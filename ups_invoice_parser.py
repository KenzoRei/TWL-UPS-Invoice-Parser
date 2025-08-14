import shutil
from pathlib import Path
import sys
from typing import List
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from models import Invoice, Shipment, Package, Charge, Location
import logging
import pickle
import csv
from utils.file_chooser import FileChooser

# General costs are cost that will NEVER allocate to any costomer.
# This is the first priority over all rules
General_Cost_EN = {
    "Payment Processing Fee"
}

# Special Customer for special handlings:
# 1) full charge of specific accounts are assigned to these customers
# 2) Exclude from all YDD import tempaltes
# 3) Total ar amount = total ap amount * index, instead of aggregation of all charges
# 4) Need split original invoices instead of re-arranged invoices
SPECIAL_CUSTOMERS = {
    "F000222": {"accounts": {"H930G2", "H930G3", "H930G4", "R1H015", "XJ3936"}},
    "F000208": {"accounts": {"HE6132"}},
}
SPECIAL_ACCT_TO_CUST = {acct: cust for cust, v in SPECIAL_CUSTOMERS.items() for acct in v["accounts"]}
SPECIAL_CUSTS = set(SPECIAL_CUSTOMERS.keys())

def is_blank(val) -> bool:
    """
    Returns True if value is considered empty:
    - NaN / None
    - Empty string ""
    - String containing only whitespace
    """
    if pd.isna(val):
        return True
    if isinstance(val, str):
        if val.strip() == "":
            return True
        if val.strip().lower() == "nan":
            return True
    return False

# -- inside UpsInvLoader --
class UpsInvLoader:
    def __init__(self, input_dir: str = "."):
        self.input_dir = Path(input_dir)
        self.invoices: list[Path] = []
        self.basic_info: list[dict] = []
        self.batch_number: str | None = None
        self._chooser = FileChooser(self.input_dir)

    def choose_files_dialog(self) -> list[Path]:
        self.invoices = self._chooser.pick_csvs(interactive=True, cli_fallback=True)
        return self.invoices

    def choose_files_cli(self) -> list[Path]:
        self.invoices = self._chooser.pick_csvs(interactive=False, cli_fallback=True)
        return self.invoices

    # --- helper: read the first row as fields (handles quotes/commas) ---
    def _read_first_row(self, file: Path) -> list[str]:
        tried_encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in tried_encs:
            try:
                with file.open("r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    return next(reader)  # first row only
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except StopIteration:
                # empty file
                return []
        # fallback with replacement to avoid crash; still try to parse
        with file.open("r", encoding="latin1", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def validate_format(self) -> bool:
        """
        Validate all selected CSVs:
        (1) no header
        (2) version == '2.1' (fatal)
        (3) 250 columns
        (4) acct# len 10, starts '0000'
        (5) inv# len 15, starts '000000' -> collects batch# (last 3)
        Additionally: ALL files must share the SAME batch number.
        (6) inv amount numeric      
        """
        if not self.invoices:
            logging.warning("No files selected for validation.")
            return False

        all_valid = True
        batch_seen: str | None = None

        for file in self.invoices:
            row = self._read_first_row(file)
            fname = file.name

            if not row:
                logging.warning(f"[{fname}] Empty file or unreadable first row.")
                all_valid = False
                continue

            # (1) header check (heuristic)
            v_raw = row[0].strip() if len(row) >= 1 else ""
            try:
                float(v_raw)
            except ValueError:
                logging.warning(f"[{fname}] Header detected or invalid version field '{v_raw}'. Files must NOT have a header.")
                all_valid = False

            # (2) version must be exactly 2.1
            if v_raw != "2.1":
                logging.error(f"[{fname}] Invoice output version must be '2.1', got '{v_raw}'.")
                all_valid = False
                # no need to check further for this file
                continue

            # (3) 250 columns
            if len(row) != 250:
                logging.warning(f"[{fname}] Column count expected 250, got {len(row)}.")
                all_valid = False  # treat as fail

            # (4) account number format
            acct = row[2].strip() if len(row) >= 3 else ""
            if not (len(acct) == 10 and acct.startswith("0000")):
                logging.warning(f"[{fname}] Account number format invalid: '{acct}' (expect 10 chars starting with '0000').")
                all_valid = False

            # (5) invoice number format + batch# extraction
            inv = row[5].strip() if len(row) >= 6 else ""
            if not (len(inv) == 15 and inv.startswith("000000")):
                logging.warning(f"[{fname}] Invoice number format invalid: '{inv}' (expect 15 chars starting with '000000').")
                all_valid = False
            else:
                bn = inv[-3:]  # proposed batch number
                if batch_seen is None:
                    batch_seen = bn
                elif bn != batch_seen:
                    logging.error(f"[{fname}] Batch number '{bn}' differs from previous '{batch_seen}'. All invoices must be in the SAME batch.")
                    all_valid = False

            # (6) invoice amount numeric
            inv_amt_raw = row[10].strip() if len(row) >= 11 else ""
            try:
                float(inv_amt_raw.replace(",", ""))
            except ValueError:
                logging.warning(f"[{fname}] Invoice amount (11th field) not numeric: '{inv_amt_raw}'.")
                all_valid = False

        # Only set self.batch_number if validation passed and we saw one
        if all_valid and batch_seen is not None:
            self.batch_number = batch_seen
        else:
            self.batch_number = None

        return all_valid

    def archive_raw_invoices(self) -> None:
        """
        Archive all selected CSV files into:
            <project_root>/data/raw_invoices[/<batch_number>]
        """
        if not self.invoices:
            raise ValueError("No files selected to archive. Run choose_files_dialog() first.")

        base_path = Path(__file__).resolve().parent / "data" / "raw_invoices"
        out_path = base_path / str(self.batch_number) if self.batch_number else base_path
        out_path.mkdir(parents=True, exist_ok=True)

        for file in self.invoices:
            shutil.copy(file, out_path / file.name)

        print(f"📁 Archived {len(self.invoices)} files to {out_path}")
    
    def run_import(
        self,
        *,
        interactive: bool = True,
        cli_fallback: bool = True
    ) -> None:
        """
        1) choose files (dialog/CLI)
        2) validate (fails if strict=True and any check fails)
        3) archive to data/raw_invoices[/batch]
        """
        # 1) choose
        if interactive:
            try:
                self.choose_files_dialog()
            except Exception:
                if not cli_fallback:
                    raise
                self.choose_files_cli()
        if not self.invoices:
            raise ValueError("No CSV files selected.")

        # 2) validate
        if not self.validate_format():
            raise ValueError("Validation failed; see logs.")

        # 3) archive
        self.archive_raw_invoices()

class UpsInvNormalizer:
    def __init__(self, file_list: List[Path]):
        """
        :param file_list: List of raw invoice CSV file paths
        """
        self.file_list = file_list
        self.header_path = Path(__file__).resolve().parent / "data" / "mappings" / "OriHeadr.csv"
        self.headers = pd.DataFrame()
        self.raw_dataframes = []
        self.normalized_df = pd.DataFrame()
        self.all_col_name = []
        self.filtered_col_name = []
        self.date_cols = []
        self.dtype_map = {}        

        # Load and process header mapping when initialized
        self._load_header_mapping()
    
    def _validate_header_mapping(self):
        """
        Validate that header mapping file for:
         (1) NaNs in critical columns.
         (2) formats in column 'Format'
        Applied in method '_load_header_mapping'
        """
        if self.headers['Column Name'].isna().any():
            print("❌ ERROR: 'Column Name' column in header mapping contains NaN values. Please fix OriHeadr.csv.")
            exit(1)
        if self.headers['Format'].isna().any():
            print("❌ ERROR: 'Format' column in header mapping contains NaN values. Please fix OriHeadr.csv.")
            exit(1)

        # Allowed formats
        allowed_formats = {'str', 'float', 'int', 'date'}
        invalid_formats = set(self.headers['Format'].unique()) - allowed_formats
        if invalid_formats:
            print(f"❌ ERROR: 'Format' column contains invalid values: {invalid_formats}. Allowed values are: {allowed_formats}. Please fix OriHeadr.csv.")
            exit(1)

    def _load_header_mapping(self):
        """Load header mapping CSV, build rename map, dtype map, and date columns."""
        self.headers = pd.read_csv(self.header_path)
        self._validate_header_mapping()
        self.all_col_name = self.headers['Column Name'].tolist()
        mask_non_date = (self.headers['Flag'] == 1) & (self.headers['Format'] != 'date')
        mask_date = (self.headers['Flag'] == 1) & (self.headers['Format'] == 'date')
        format_map = {
            'str': str,
            'float': float,
            'int': int
            }
        self.dtype_map = dict(zip(
            self.headers.loc[mask_non_date, 'Column Name'],
            self.headers.loc[mask_non_date, 'Format'].map(format_map)
            ))
        self.filtered_col_name = self.headers.loc[self.headers['Flag'] == 1,'Column Name'].tolist()
        self.date_cols = self.headers.loc[mask_date, 'Column Name'].tolist()
    
    def _save_trk_nums(self):
        out_path = Path(__file__).resolve().parent / "output"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "trk_nums.csv"

        # setup account list for special customers
        special_cust_acct = [
            acct
            for cust in SPECIAL_CUSTOMERS.values()
            for acct in cust["accounts"]
            ]

        # filter all unique tracking numbers
        mask_trk_num = ~self.normalized_df["Account Number"].isin(special_cust_acct)
        trk_num = (
            self.normalized_df.loc[mask_trk_num, "Tracking Number"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .drop_duplicates()
            .sort_values()
        )
        trk_num.to_csv(out_file, index=False)

    def load_invoices(self):
        """Load CSV invoice files with correct dtypes and dates."""
        self.raw_dataframes = []
        tried_encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        for file in self.file_list:
            last_err = None
            for enc in tried_encs:
                try:
                    df = pd.read_csv(
                        file,
                        header=None,
                        names=self.all_col_name,
                        dtype=self.dtype_map,
                        encoding=enc,                 # ← try encodings
                        encoding_errors="strict",     # or "replace" to keep going
                    )
                    # success -> stop trying encodings
                    print(f"✓ Loaded {file.name} with encoding {enc}")
                    break
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
            else:
                # no encoding worked; as a last resort, force replacement
                df = pd.read_csv(
                    file,
                    header=None,
                    names=self.all_col_name,
                    dtype=self.dtype_map,
                    encoding="latin1",
                    encoding_errors="replace",  # keeps data, replaces bad bytes
                )
                print(f"! Loaded {file.name} with latin1 (replacement used)")

            # keep only flagged columns
            df = df.loc[:, self.filtered_col_name]

            # parse date columns safely (mixed formats)
            for date_col in self.date_cols:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

            self.raw_dataframes.append(df)

    def merge_invoices(self):
        """Merge loaded DataFrames into one."""
        self.normalized_df = pd.concat(self.raw_dataframes, ignore_index=True)

    def standardize_invoices(self):
        """Standardize and enrich columns as per business rules."""

        # to confirm string columns are converted correctly
        str_cols = self.headers.loc[(self.headers['Flag'] == 1) & (self.headers['Format'] == 'str'), 'Column Name'].tolist()
        for col in str_cols:
            self.normalized_df[col] = self.normalized_df[col].astype(str)

        df = self.normalized_df        

        # Convert to datetime after columns are renamed
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

        df['Account Number'] = df['Account Number'].astype(str).str[-6:]
        df['Invoice Number'] = df['Invoice Number'].astype(str).str[-9:]

        # Divide "dimension" into 3 columns
        df['Package Dimensions'] = df['Package Dimensions'].str.replace(' ', '', regex=False)
        dims = df['Package Dimensions'].str.split('x', expand=True).astype(float)
        dims.columns = ['Billed Length', 'Billed Width', 'Billed Height']
        idx = df.columns.get_loc('Package Dimensions') + 1
        for i, col in enumerate(dims.columns):
            df.insert(idx + i, col, dims[col])
        # Divide entered dimension into 3 columns
        df['Place Holder 35'] = df['Place Holder 35'].str.replace(' ', '', regex=False)
        dims2 = df['Place Holder 35'].str.split('x', expand=True).astype(float)
        dims2.columns = ['Entered Length', 'Entered Width', 'Entered Height']
        idx2 = df.columns.get_loc('Place Holder 35') + 1
        for i, col in enumerate(dims2.columns):
            df.insert(idx2 + i, col, dims2[col])

        insert_idx = df.columns.get_loc('Incentive Amount')
        df.insert(insert_idx, 'Basis Amount', df['Incentive Amount'] + df['Net Amount'])

        charge_idx = df.columns.get_loc('Charge Description') + 1
        df.insert(charge_idx, 'Charge_Cate_EN', '')
        df.insert(charge_idx + 1, 'Charge_Cate_CN', '')

        self.normalized_df = df
        self._save_trk_nums()

    def get_normalized_data(self) -> pd.DataFrame:
        return self.normalized_df

class UpsCustomerMatcher:
    def __init__(self, normalized_df: pd.DataFrame, mapping_file: Path | None = None):
        """
        :param normalized_df: DataFrame output from normalizer.get_normalized_data()
        :param mapping_file:  Optional path to 数据列表*.xlsx; can be chosen later via choose_mapping_file_dialog()
        """
        self.df = normalized_df.copy()  # Keep original safe
        if "cust_id" not in self.df.columns:  # Add empty column if not exist
            self.df["cust_id"] = ""
        self.df["Charge_Cate_EN"] = ""
        self.df["Charge_Cate_CN"] = ""
        self.DEFAULT_CUST_ID = "F000999"

        self.base_path = Path(__file__).resolve().parent

        # --- single mapping file (user-provided) ---
        self.mapping_file: Path | None = mapping_file
        self.mapping_cust_df = pd.DataFrame()

        # Other mapping paths (unchanged)
        self.mapping_pickup = self.base_path / "data" / "mappings" / "Pickups.csv"   # for daily pickup mapping
        self.mapping_pickup_df = pd.DataFrame()
        self.mapping_chrg = self.base_path / "data" / "mappings" / "Charges.csv"     # for charge category mapping
        self.mapping_chrg_df = pd.DataFrame()
        self.mapping_ar = self.base_path / "data" / "mappings" / "ARCalculator.csv"  # for ar amount amplifier & modifier flag
        self.mapping_ar_df = pd.DataFrame()

        # Lookup dicts
        self.trk_to_cust: dict[str, tuple[str, str]] = {}
        self.dict_chrg = {}
        self.dict_pickup = {}
        self.dict_ar = {}

        # cust.id that will be excluded from exception imports (since they don't need to be preserved in YDD)
        self.excluded_from_exception = list(SPECIAL_CUSTS)

        # NOTE: we no longer auto-load here, because the user might not have produced the file yet.
        # Call choose_mapping_file_dialog() or set_mapping_file(), then match_customers() will load.

    # -------- file selection (single file) --------
    def set_mapping_file(self, path: Path) -> None:
        """Programmatically set the 数据列表*.xlsx path."""
        self.mapping_file = Path(path) if path else None

    def choose_mapping_file_dialog(self) -> Path | None:
        """Let the user pick exactly one mapping file via GUI."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise RuntimeError("tkinter is not available for file selection.") from e

        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select mapping file (数据列表*.xlsx)",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        root.update(); root.destroy()

        self.mapping_file = Path(file_path) if file_path else None
        return self.mapping_file

    # -------- loading mapping assets --------
    def _load_mapping(self) -> None:
        """Load all mappings. Requires self.mapping_file to be set."""
        if self.mapping_file is None:
            raise FileNotFoundError(
                "Mapping file not set. Call choose_mapping_file_dialog() or set_mapping_file(Path)."
            )

        # 1) Load 数据列表*.xlsx into self.mapping_cust_df and build trk→(客户编号, 转单号)
        self.mapping_cust_df = pd.read_excel(self.mapping_file)
        required_cols = {"子转单号", "客户编号", "转单号"}
        if not required_cols.issubset(self.mapping_cust_df.columns):
            missing = required_cols - set(self.mapping_cust_df.columns)
            raise ValueError(f"Mapping file missing required columns: {missing}")
        self._build_lookup_dict()

        # 2) Charges.csv → dict
        self.mapping_chrg_df = pd.read_csv(self.mapping_chrg)
        # Assumes first column is the key (Charge Description)
        self.dict_chrg = self.mapping_chrg_df.set_index(self.mapping_chrg_df.columns[0]).to_dict(orient="index")

        # 3) Pickups.csv → dict
        self.mapping_pickup_df = pd.read_csv(self.mapping_pickup)
        self.dict_pickup = self.mapping_pickup_df.set_index(self.mapping_pickup_df.columns[0]).to_dict(orient="index")

        # 4) ARCalculator.csv → dict
        self.mapping_ar_df = pd.read_csv(self.mapping_ar)
        self.dict_ar = self.mapping_ar_df.set_index(self.mapping_ar_df.columns[0]).to_dict(orient="index")

    def _build_lookup_dict(self) -> None:
        """Flatten mapping and build dictionary: {Tracking Number: (客户编号, 转单号)} using vectorized explode."""
        df = self.mapping_cust_df.copy()
        # Split "子转单号" by comma into multiple rows
        df["子转单号"] = (
            df["子转单号"]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        df = df.assign(Tracking=df["子转单号"].str.split(",")).explode("Tracking")
        df["Tracking"] = df["Tracking"].astype(str).str.replace(" ", "", regex=False)
        df = df[df["Tracking"] != ""]  # drop blanks

        self.trk_to_cust = dict(
            zip(df["Tracking"], zip(df["客户编号"].astype(str), df["转单号"].astype(str)))
        )

    # -------- main workflow --------
    def match_customers(self) -> None:
        """Run classification, matching, AR calculation, exception template, and unmapped charge logging."""
        # Ensure mappings are loaded (user may have chosen the file just now)
        if not self.trk_to_cust or not self.dict_chrg or not self.dict_pickup or not self.dict_ar:
            self._load_mapping()

        # 0) Pre-assign specials by account (highest priority, vectorized)
        self.df["cust_id"] = self.df["cust_id"].astype(str)
        self.df["cust_id_special"] = self.df["Account Number"].map(SPECIAL_ACCT_TO_CUST)
        mask_special = self.df["cust_id_special"].notna()
        self.df.loc[mask_special, "cust_id"] = self.df.loc[mask_special, "cust_id_special"]
        self.df.drop(columns=["cust_id_special"], inplace=True)

        """
        1. Match and overwrite cust_id and Lead Shipment Number in self.df
        2. Generate YiDiDa import template for unmatched charges
        3. Calculate AR amount
        """
        exception_rows = []
        for idx, row in self.df.iterrows():
            cust_id, lead_shipment = np.nan, np.nan
            exception_flag = False

            # classify charges
            category_en, category_cn = self._charge_classifier(row)
            self.df.at[idx, "Charge_Cate_EN"] = category_en
            self.df.at[idx, "Charge_Cate_CN"] = category_cn

            # matchup cust.id
            trk_num = str(row["Tracking Number"])
            if trk_num in self.trk_to_cust:
                cust_id, lead_shipment = self.trk_to_cust[trk_num]
            else:
                cust_id = self._exception_handler(self.df.loc[idx])
                exception_flag = True
                lead_shipment = row["Lead Shipment Number"] if not is_blank(row["Lead Shipment Number"]) else trk_num

            self.df.at[idx, "cust_id"] = cust_id
            # REMINDER: overwrite lead shipment# and trk# may lead to info loss!!
            self.df.at[idx, "Lead Shipment Number"] = lead_shipment
            if is_blank(row["Tracking Number"]):
                self.df.at[idx, "Tracking Number"] = lead_shipment
            if is_blank(self.df.at[idx, "Lead Shipment Number"]) and self.df.at[idx, "cust_id"] != "F000999":
                ls = row["Invoice Number"] + "-AcctExpense"
                self.df.at[idx, "Lead Shipment Number"] = ls
                self.df.at[idx, "Tracking Number"] = ls
                self.df.at[idx, "Shipment Reference Number 1"] = ls

            if exception_flag and not is_blank(row["Lead Shipment Number"]):
                exception_rows.append(self.df.loc[idx].to_dict())

        self._ar_calculator()

        # Exception template
        df_exception = pd.DataFrame(exception_rows)
        num_cols = ["总实重", "长", "宽", "高"]
        for col in df_exception.columns:
            if col in num_cols:
                df_exception[col] = pd.to_numeric(df_exception[col], errors="coerce").fillna(0)
            else:
                df_exception[col] = df_exception[col].replace({np.nan: ""}).replace({"nan": ""})
        self._template_generator(df_exception)

        # Save all rows with undefined charge categories
        unmapped_charges = self.df[self.df["Charge_Cate_EN"] == "Not Defined Charge"]
        if not unmapped_charges.empty:
            output_path = self.base_path / "output" / "UnmappedCharges.xlsx"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            unmapped_charges.to_excel(output_path, index=False)

    def _exception_handler(self, row: pd.Series) -> str:
        """Manually match a shipment with cust_id and return it."""

        # 0. General cost 
        if row["Charge_Cate_EN"] in General_Cost_EN:
            return self.DEFAULT_CUST_ID

        # 1. pass cust_id if it already exists
        if not is_blank(row["cust_id"]) and row["cust_id"] != self.DEFAULT_CUST_ID:
            return row["cust_id"]
        
        # 2. Import department pickup
        elif (
            row["Lead Shipment Number"].startswith("2") and
            "import" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000223"

        # 3. UPEX (vermilion match)
        elif (
            "vermilion" in row["Sender Name"].lower() or
            "vermilion" in row["Sender Company Name"].lower() or
            "vermilion" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000215"

        # 4. Bondex (pickup fee allocates to TWW)
        elif row["Account Number"] in ["K5811C", "F03A44"]:
            if row["Charge_Cate_EN"] in ["Daily Pickup", "Daily Pickup - Fuel"]:
                return self.DEFAULT_CUST_ID
            else:
                return "F000281"

        # 5. SPT
        elif row["Account Number"] == "HE6132":
            return "F000208"

        # 6. Transware
        elif row["Account Number"] in (["H930G2", "H930G3", "H930G4", "R1H015", "XJ3936"]):
            return "F000222"
        elif not is_blank(row["Tracking Number"]) or not is_blank(row["Lead Shipment Number"]):
            return "F000222"

        # 7. Daily Pickup logic
        elif row["Charge_Cate_EN"] in ["Daily Pickup", "Daily Pickup - Fuel"]:
            return self.dict_pickup.get(row["Account Number"], {}).get("Cust.ID", self.DEFAULT_CUST_ID)

        # 8. Generic cost rules
        elif str(row["Charge_Cate_EN"]).upper() in ["SCC AUDIT FEE", "POD FEE"]:
            return self.DEFAULT_CUST_ID

        # 9. Fallback
        return self.DEFAULT_CUST_ID

    
    def _charge_classifier(self, row: pd.Series) -> tuple[str, str]:
        """
        Classify charge by "charge description" and all types of class codes
        return category in both English and Chinese version
        """

        ChrgDesc = row["Charge Description"]
        Chrg_EN = "Not Defined Charge"
        Chrg_CN = "未分类费用"
        for term in [
            "Not Previously Billed ",
            "Void ",
            "Shipping Charge Correction ",
            " Adjustment",
            "ZONE ADJUSTMENT "
            ]:
            ChrgDesc = ChrgDesc.replace(term, "")
        
        # IMPORTANT: CHARGE CLASSIFICATION LOGIC HERE!!!
        if is_blank(row["Tracking Number"]) and row["Charge Category Detail Code"] == "SVCH":
            if row["Charge Description"] == "Payment Processing Fee":
                Chrg_EN = "Payment Processing Fee"
                Chrg_CN = "信用卡手续费"
            elif row["Charge Description"] == "Fuel Surcharge":
                Chrg_EN = "Daily Pickup - Fuel"
                Chrg_CN = "每日取件-燃油"
            elif row["Charge Description"] == "Service Charge":
                Chrg_EN = "Daily Pickup"
                Chrg_CN = "每日取件"
        elif ChrgDesc in self.dict_chrg:
            Chrg_EN = self.dict_chrg[ChrgDesc]["Charge_Cate_EN"]
            Chrg_CN = self.dict_chrg[ChrgDesc]["Charge_Cate_CN"]
        elif row["Charge Category Detail Code"] == "CADJ" and row["Charge Classification Code"] == "MSC":
            if "audit fee" in row["Miscellaneous Line 1"].lower():
                Chrg_EN = "SCC Audit Fee"
                Chrg_CN = "UPS SCC审计费"
            else:
                Chrg_EN = "Shipping Charge Correction"
                Chrg_CN = "费用修正"
        elif row["Charge Category Detail Code"] == "FPOD":
            Chrg_EN = "POD Fee"
            Chrg_CN = "送抵证明"
        
        return Chrg_EN, Chrg_CN
        
    def _template_generator(self, df: pd.DataFrame):
        """ Generate an YiDiDa template and save is to output folder"""
        # TODO: finish template generator
        cols_to_check = ["Lead Shipment Number", "Tracking Number"]
        condition_missing = df[cols_to_check].apply(lambda row: any(is_blank(val) for val in row), axis=1)
        condition_exclude = ~df["cust_id"].isin(self.excluded_from_exception)
        df_exceptions = df[~condition_missing & condition_exclude]
        df_exceptions = df_exceptions.drop_duplicates(subset=["Lead Shipment Number", "Tracking Number"], keep="first")
        col_names = ["客户代码", "转单号", "承运商子单号", "客户单号(文本格式)", 
                     "收货渠道", "目的地国家", "件数", "总实重", "长", "宽", 
                     "高", "省份/洲名简码", "城市", "邮编", "收件公司", 
                     "收件人姓名", "收件人电话", "收件人地址一", "收件人地址二", 
                     "包裹类型", "报关方式", "付税金", "中文品名", "英文品名", 
                     "海关编码", "数量", "币种", "单价", "总价", "购买保险", 
                     "寄件人姓名", "寄件公司", "寄件电话", "寄件人地址一", "寄件人地址二", 
                     "寄件人城市", "寄件人州名", "寄件人国家", "寄件人邮编", 
                     "货物特性", "收货备注", "预报备注"
        ]
        output_template = pd.DataFrame(columns=col_names)
        output_template["客户代码"] = df_exceptions["cust_id"].fillna("F000999").astype(str)
        output_template["转单号"] = df_exceptions["Lead Shipment Number"].fillna("UNKNOWN").astype(str)
        output_template["承运商子单号"] = df_exceptions["Tracking Number"].fillna("UNKNOWN").astype(str)
        output_template["客户单号(文本格式)"] = (
            df_exceptions["Shipment Reference Number 1"].fillna("NoRef").astype(str) + "-" +
            df_exceptions["Lead Shipment Number"].fillna("UNKNOWN").astype(str)
        ).str[:34]        
        output_template["收货渠道"] = "UPS Exception Handling"
        output_template["目的地国家"] = "US"
        output_template["件数"] = 1
        output_template["总实重"] = (
            pd.to_numeric(df_exceptions["Billed Weight"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x / 2.204, 1))
        ).round(2)
        output_template["长"] = (
            pd.to_numeric(df_exceptions["Billed Length"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["宽"] = (
            pd.to_numeric(df_exceptions["Billed Width"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["高"] = (
            pd.to_numeric(df_exceptions["Billed Height"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["省份/洲名简码"] = df_exceptions["Receiver State"].replace("", "CA").astype(str)
        output_template["城市"] = df_exceptions["Receiver City"].replace("", "UNKNOWN").astype(str)
        output_template["邮编"] = df_exceptions["Receiver Postal"].replace("", "90248").astype(str)
        output_template["收件公司"] = df_exceptions["Receiver Company Name"].replace("", "UNKNOWN").astype(str)
        output_template["收件人姓名"] = df_exceptions["Receiver Name"].replace("", "UNKNOWN").astype(str)
        output_template["收件人地址一"] = df_exceptions["Receiver Address Line 1"].replace("", "UNKNOWN").astype(str)
        output_template["收件人地址二"] = df_exceptions["Receiver Address Line 2"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人姓名"] = df_exceptions["Sender Name"].replace("", "UNKNOWN").astype(str)
        output_template["寄件公司"] = df_exceptions["Sender Company Name"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人地址一"] = df_exceptions["Sender Address Line 1"].fillna("UNKNOWN").astype(str)
        output_template["寄件人地址二"] = df_exceptions["Sender Address Line 2"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人城市"] = df_exceptions["Sender City"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人州名"] = df_exceptions["Sender State"].replace("", "CA").astype(str)
        output_template["寄件人国家"] = "US"
        output_template["寄件人邮编"] = df_exceptions["Sender Postal"].replace("", "90248").astype(str)
        output_template["预报备注"] = "ManualImport " + datetime.date.today().strftime("%Y-%m-%d")
        for col in ["报关方式", "付税金", "中文品名", "英文品名", "海关编码", 
            "数量", "币种", "单价", "总价", "购买保险", "货物特性", "收货备注"]:
            output_template[col] = ""
        if not output_template.empty:
            output_path = self.base_path / "output" / "ExceptionImport_YDD.xlsx"
            output_path.parent.mkdir(parents=True, exist_ok=True) 
            output_template = output_template.fillna("").replace("nan", "")
            output_template.to_excel(output_path, index=False)
        print(f"✅ YDD Exception Template generated with {len(output_template)} rows.")
        print(f"📁 YDD Exception Template Saved to: {output_path}")

    def _ar_calculator(self):
        """Calculate ar amount according to cust_id"""
        # verify self.df
        empty_cust_id_rows = self.df[self.df["cust_id"].apply(is_blank)]
        if not empty_cust_id_rows.empty:
            print(f"[Warning] {len(empty_cust_id_rows)} rows have empty cust_id.") # log output?
        invalid_cust_id_rows = self.df[~self.df["cust_id"].isin(self.dict_ar.keys())]
        if not invalid_cust_id_rows.empty:
            print(f"[Warning] {len(invalid_cust_id_rows)} rows have unmapped cust_id (not in AR mapping).") # log output?
        
        # Calculate AR Amount using business rules and AR factor mapping.
        # Extract mapping components as Series via dict lookup
        self.df["AR_Factor"] = self.df["cust_id"].map(lambda cid: self.dict_ar.get(cid, {}).get("Factor", 0.0))
        self.df["Flag_Modifier"] = self.df["cust_id"].map(lambda cid: self.dict_ar.get(cid, {}).get("Flag_Modifier", ""))

        # Business rule — if SIM + negative + no modifier → AR = 0
        cond_special_zero = (
            (self.df["Charge_Cate_EN"] == "Special Incentive Modifier") &
            (self.df["Net Amount"] < 0) &
            (self.df["Flag_Modifier"].apply(is_blank))
        )

        # Default AR amount = Net × Factor
        self.df["AR_Amount"] = (self.df["Net Amount"] * self.df["AR_Factor"]).round(2)

        # Apply override where condition is met
        self.df.loc[cond_special_zero, "AR_Amount"] = 0.00

    def get_matched_data(self) -> pd.DataFrame:
        """Return the updated DataFrame with customer info matched."""
        return self.df    

class UpsInvoiceBuilder:
    def __init__(self, normalized_df: pd.DataFrame):
        self.df = normalized_df
        self.output_path = Path(__file__).resolve().parent / "data" / "raw_invoices"
        self.invoices: dict[str, Invoice] = {}
        # dict to store scc pkgs where shipment_trk_num and inv_num were kept in tuple
        self.scc_packages: dict[str, tuple[str, str]] = {}
        self.scc_unit_charge = 1.65

    def _parse_date(self, val):
        if pd.isna(val):
            return None
        return pd.to_datetime(val).date()

    def build_invoices(self):
        """Convert normalized DataFrame into nested Invoice → Shipment → Package → Charge structure."""

        # verify headers
        missing_cols = self._verify_invoice(self.df)
        if missing_cols != []:
            missing_cols_list = ",".join(missing_cols)
            # TODO: improve warning msg/log msg
            logging.warning("Missing columns: %s", missing_cols_list)
        
        # grouping & object creation logic
        for _, row in self.df.iterrows():
            
            inv_num = row["Invoice Number"]
            if inv_num not in self.invoices:
                # create invoice info
                invoice = Invoice()
                invoice.carrier = "UPS"
                invoice.inv_date = self._parse_date(row["Invoice Date"])
                invoice.inv_num = row["Invoice Number"]
                invoice.acct_num = row["Account Number"]
                invoice.batch_num = invoice.inv_num[-3:]
                self.invoices[inv_num] = invoice

            # for general invoice cost
            if is_blank(row["Lead Shipment Number"]):
                self._build_invoice_cost(row, invoice)
            
            # add/update shipment info
            else:
                self._build_shipment(row, invoice)

    def _verify_invoice(self, df: pd.DataFrame) -> list:
        missing_cols = []
        col_names = ["Account Number", "Invoice Date", "Invoice Number", 
                     "Invoice Currency Code", "Invoice Amount", 
                     "Transaction Date", "Lead Shipment Number", 
                     "Shipment Reference Number 1", "Shipment Reference Number 2", 
                     "Tracking Number", "Package Reference Number 1", 
                     "Package Reference Number 2", "Entered Weight", 
                     "Billed Weight", "Billed Weight Type", "Billed Length", 
                     "Billed Width", "Billed Height", "Zone", "Charge_Cate_EN", 
                     "Charge_Cate_CN", "Charged Unit Quantity", 
                     "Transaction Currency Code", "Basis Amount", 
                     "Incentive Amount", "Net Amount", "Sender Name", 
                     "Sender Company Name", "Sender Address Line 1", 
                     "Sender Address Line 2", "Sender City", "Sender State", 
                     "Sender Postal", "Sender Country", "Receiver Name", 
                     "Receiver Company Name", "Receiver Address Line 1", 
                     "Receiver Address Line 2", "Receiver City", "Receiver State", 
                     "Receiver Postal", "Receiver Country", "Miscellaneous Line 1", 
                     "Miscellaneous Line 2", "Miscellaneous Line 3", 
                     "Miscellaneous Line 4", "Miscellaneous Line 5", 
                     "Original Shipment Package Quantity", "Entered Length", 
                     "Entered Width", "Entered Height", "cust_id", "AR_Amount"]
        for col_name in col_names:
            if col_name not in df.columns:
                missing_cols.append(col_name)
        return missing_cols

    def _build_shipment(self, row: pd.Series, invoice: Invoice):
        Lead_Shipment_Num = row["Lead Shipment Number"]
        if Lead_Shipment_Num not in invoice.shipments:
            # create shipment info
            invoice.shipments[Lead_Shipment_Num] = Shipment()
            shipment = invoice.shipments[Lead_Shipment_Num]
            shipment.main_trk_num = Lead_Shipment_Num
            shipment.cust_id = row["cust_id"]
            shipment.tran_date= self._parse_date(row["Transaction Date"])
            shipment.zone = row["Zone"]
            shipment.ship_ref1 = row["Shipment Reference Number 1"]
            shipment.ship_ref2 = row["Shipment Reference Number 2"]
            self._build_location(row, shipment)
        else:
            shipment = invoice.shipments[Lead_Shipment_Num]

        if is_blank(row["Tracking Number"]):
            self._build_shipment_cost(row, shipment, invoice)
        # add/update package info
        else:
            self._build_package(row, shipment, invoice)

    def _build_package(self, row: pd.Series, shipment: Shipment, invoice: Invoice):
        # reminder: when creating a pkg, need to add 1 pkg at shipment lvl
        pkg_trk_num = row["Tracking Number"]
        if pkg_trk_num not in shipment.packages:
            shipment.packages[pkg_trk_num] = Package()
            package = shipment.packages[pkg_trk_num]
            package.trk_num = pkg_trk_num

            package.entered_wgt = row["Entered Weight"]
            package.billed_wgt = row["Billed Weight"]
            # add same wgt to shipment lvl:
            shipment.entered_wgt += row["Entered Weight"]
            shipment.billed_wgt += row["Billed Weight"]

            package.length = row["Billed Length"]
            package.width = row["Billed Width"]
            package.height = row["Billed Height"]

            package.pkg_ref1 = row["Package Reference Number 1"]
            package.pkg_ref2 = row["Package Reference Number 2"]

        else:
            package = shipment.packages[pkg_trk_num]

        # update charge at pkg lvl
        self._build_package_charge(row, package, shipment, invoice)

        # update SCC flag at pkg lvl
        # SCC rule:
        # 1. pkg dim is not empty
        # 2. Charge Category Detail Code is "SCC"
        if not is_blank(row["Package Dimensions"]) and row["Charge Category Detail Code"] == "SCC":
            package.flag_UPS_SCC = True
            self.scc_packages[package.trk_num] = (shipment.main_trk_num, invoice.inv_num)

    def _build_invoice_cost(self, row: pd.Series, invoice: Invoice): 
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        if charge_cate not in invoice.inv_charge:
            invoice.inv_charge[charge_cate] = Charge()
            invoice_charge_detail = invoice.inv_charge[charge_cate]
            invoice_charge_detail.charge_en = row["Charge_Cate_EN"]
            invoice_charge_detail.charge_cn = row["Charge_Cate_CN"]
            invoice_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            invoice_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        invoice_charge_detail = invoice.inv_charge[charge_cate]
        invoice_charge_detail.ap_amt += ap_amt
        invoice_charge_detail.inc_amt += inc_amt
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt           

    def _build_shipment_cost(self, row: pd.Series, shipment: Shipment, invoice: Invoice): 
        # reminder: when adding a charge, need to add same amt at invoice&shipment lvl
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        if charge_cate not in shipment.shipment_charge:
            shipment.shipment_charge[charge_cate] = Charge()
            shipment_charge_detail = shipment.shipment_charge[charge_cate]
            shipment_charge_detail.charge_en = row["Charge_Cate_EN"]
            shipment_charge_detail.charge_cn = row["Charge_Cate_CN"]
            shipment_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            shipment_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        # update shipment general charge
        shipment_charge_detail = shipment.shipment_charge[charge_cate]
        shipment_charge_detail.ap_amt += ap_amt
        shipment_charge_detail.ar_amt += ar_amt
        shipment_charge_detail.inc_amt += inc_amt
        # amt aggr@shipment lvl
        shipment.ap_amt += ap_amt
        shipment.ar_amt += ar_amt
        # amt aggr@invoice lvl
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt

    def _build_package_charge(self, row: pd.Series, package: Package, shipment: Shipment, invoice: Invoice):
        # reminder: when adding a charge, need to add same amt at invoice&shipment lvl
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        if charge_cate not in package.charge_detail:
            package.charge_detail[charge_cate] = Charge()
            package_charge_detail = package.charge_detail[charge_cate]
            package_charge_detail.charge_en = row["Charge_Cate_EN"]
            package_charge_detail.charge_cn = row["Charge_Cate_CN"]
            package_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            package_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        # update pkg charge
        package_charge_detail = package.charge_detail[charge_cate]
        package_charge_detail.ap_amt += ap_amt
        package_charge_detail.ar_amt += ar_amt
        package_charge_detail.inc_amt += inc_amt
        # amt aggr@shipment lvl
        shipment.ap_amt += ap_amt
        shipment.ar_amt += ar_amt
        # amt aggr@invoice lvl
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt

    def _build_location(self, row: pd.Series, shipment: Shipment):
        # check if sender addr info is empty
        if is_blank(shipment.sender.zipcode):
            addr_sender = shipment.sender
            addr_sender.company = row["Sender Company Name"]
            addr_sender.contact = row["Sender Name"]
            addr_sender.addr1 = row["Sender Address Line 1"]
            addr_sender.addr2 = row["Sender Address Line 2"]
            addr_sender.city = row["Sender City"]
            addr_sender.state = row["Sender State"]
            addr_sender.zipcode = row["Sender Postal"]
            addr_sender.country = row["Sender Country"]
        
        if is_blank(shipment.consignee.zipcode):
            addr_consignee = shipment.consignee
            addr_consignee.company = row["Receiver Company Name"]
            addr_consignee.contact = row["Receiver Name"]
            addr_consignee.addr1 = row["Receiver Address Line 1"]
            addr_consignee.addr2 = row["Receiver Address Line 2"]
            addr_consignee.city = row["Receiver City"]
            addr_consignee.state = row["Receiver State"]
            addr_consignee.zipcode = row["Receiver Postal"]
            addr_consignee.country = row["Receiver Country"]

    def _scc_handler(self):
        """Calculate shipment charge correction fee by SCC flag."""
        for pkg_num, (ship_num, inv_num) in self.scc_packages.items():
            invoice = self.invoices[inv_num]
            shipment = invoice.shipments[ship_num]
            package = shipment.packages[pkg_num]

            # Check if invoice has SCC Audit Fee and it’s positive
            if "SCC Audit Fee" in invoice.inv_charge and invoice.inv_charge["SCC Audit Fee"].ap_amt > 0:
                # Deduct from invoice-level fee
                invoice.inv_charge["SCC Audit Fee"].ap_amt -= self.scc_unit_charge

                # Ensure package-level SCC Audit Fee exists
                if "SCC Audit Fee" not in package.charge_detail:
                    package.charge_detail["SCC Audit Fee"] = Charge()
                    pkg_scc_charge = package.charge_detail["SCC Audit Fee"]
                    pkg_scc_charge.charge_en = "SCC Audit Fee"
                    pkg_scc_charge.charge_cn = "UPS SCC审计费"
                else:
                    pkg_scc_charge = package.charge_detail["SCC Audit Fee"]

                # Add the SCC fee at package level
                pkg_scc_charge.ap_amt += self.scc_unit_charge
                pkg_scc_charge.ar_amt += self.scc_unit_charge

                # Adjust SCC fee at shipment level
                shipment.ap_amt += self.scc_unit_charge
                shipment.ar_amt += self.scc_unit_charge

    def save_invoices(self):
        """
        Save the composite invoice structure to a .pkl file inside:
        self.output_path / batch_number
        """
        if not self.invoices:
            print("❌ No invoices to save. Please run build_invoices() first.")
            return

        # Get batch number from any invoice
        first_invoice = next(iter(self.invoices.values()))
        batch_number = getattr(first_invoice, "batch_num", "unknown_batch")

        # Prepare folder
        batch_folder = self.output_path / batch_number
        batch_folder.mkdir(parents=True, exist_ok=True)

        # File path
        file_path = batch_folder / f"invoices_{batch_number}.pkl"

        # Save pickle
        with open(file_path, "wb") as f:
            pickle.dump(self.invoices, f)

        print(f"📁 Invoices saved to {file_path}")

    def load_invoices(self, batch_number: str):
        """
        Load previously saved invoices from a given batch number folder.
        """
        file_path = self.output_path / batch_number / f"invoices_{batch_number}.pkl"
        if not file_path.exists():
            print(f"❌ No saved invoices found at {file_path}")
            return

        with open(file_path, "rb") as f:
            self.invoices = pickle.load(f)

        print(f"✅ Invoices loaded from {file_path}")
    
    def get_invoices(self) -> dict[str, Invoice]:
        """Return all constructed Invoice objects."""
        return self.invoices

class UpsInvoiceExporter:
    GENERAL_ITEMCODE_MAP = {
        "Payment Processing Fee": "7154",
        "SCC Audit Fee": "7152",
        "Daily Pickup": "7151",
        "Daily Pickup - Fuel": "7151"
    }

    def __init__(self, invoices: dict):
        # Store composite (Invoice → Shipment → Package → Charge)
        self.invoices = invoices
        self.flat_charges = pd.DataFrame()
        self.flat_packages = pd.DataFrame()

        # Paths & batch metadata
        base = Path(__file__).resolve().parent
        self.output_path = base / "output"
        first_invoice = next(iter(self.invoices.values()))
        self.batch_number = getattr(first_invoice, "batch_num", "unknown_batch")
        self.inv_date = pd.to_datetime(getattr(first_invoice, "inv_date", None)) if getattr(first_invoice, "inv_date", None) else pd.Timestamp.today().normalize()
        self.output_path = self.output_path / self.batch_number
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.raw_path = base / "data" / "raw_invoices"

        # Mappings
        self._load_mappings()

    # ----------------------
    # Mapping loader
    # ----------------------
    def _load_mappings(self):
        """Load Xero Contacts and latest InventoryItems (for *ItemCode accounts)."""
        mappings_path = Path(__file__).resolve().parent / "data" / "mappings"

        # Contacts.csv
        contacts_file = mappings_path / "Contacts.csv"
        if not contacts_file.exists():
            raise FileNotFoundError(f"❌ Contacts.csv not found in {mappings_path}")
        self.xero_contacts = pd.read_csv(contacts_file, dtype=str).fillna("")
        print(f"✅ Loaded Contacts.csv ({len(self.xero_contacts)} rows)")

        # ARCalculator.csv
        ARCalculator_file = mappings_path / "ARCalculator.csv"
        mapping_ar_df = pd.read_csv(ARCalculator_file)
        self.dict_ar = mapping_ar_df.set_index(mapping_ar_df.columns[0]).to_dict(orient="index")

        self.dict_contacts = {}
        cols = self.xero_contacts.columns
        contact_name_col = "*ContactName" if "*ContactName" in cols else "ContactName"
        for _, row in self.xero_contacts.iterrows():
            acct = str(row.get("AccountNumber", "")).strip()
            self.dict_contacts[acct] = {
                "ContactName": str(row.get(contact_name_col, "")).strip(),
                "EmailAddress": str(row.get("EmailAddress", "")).strip(),
                "POAddress": {
                    "AttentionTo": str(row.get("POAttentionTo", "")).strip(),
                    "Line1": str(row.get("POAddressLine1", "")).strip(),
                    "Line2": str(row.get("POAddressLine2", "")).strip(),
                    "Line3": str(row.get("POAddressLine3", "")).strip(),
                    "Line4": str(row.get("POAddressLine4", "")).strip(),
                    "City": str(row.get("POCity", "")).strip(),
                    "Region": str(row.get("PORegion", "")).strip(),
                    "ZipCode": str(row.get("POZipCode", "")).strip(),
                    "Country": str(row.get("POCountry", "")).strip()
                },
                "SAAddress": {
                    "AttentionTo": str(row.get("SAAttentionTo", "")).strip(),
                    "Line1": str(row.get("SAAddressLine1", "")).strip(),
                    "Line2": str(row.get("SAAddressLine2", "")).strip(),
                    "Line3": str(row.get("SAAddressLine3", "")).strip(),
                    "Line4": str(row.get("SAAddressLine4", "")).strip(),
                    "City": str(row.get("SACity", "")).strip(),
                    "Region": str(row.get("SARegion", "")).strip(),
                    "ZipCode": str(row.get("SAZipCode", "")).strip(),
                    "Country": str(row.get("SACountry", "")).strip()
                },
                "PhoneNumber": str(row.get("PhoneNumber", "")).strip(),
                "MobileNumber": str(row.get("MobileNumber", "")).strip()
            }

        # InventoryItems-*.csv (latest)
        inventory_files = sorted(mappings_path.glob("InventoryItems-*.csv"), reverse=True)
        if not inventory_files:
            raise FileNotFoundError(f"❌ No InventoryItems-*.csv found in {mappings_path}")
        latest_inventory_file = inventory_files[0]
        self.xero_inventory = pd.read_csv(latest_inventory_file, dtype=str).fillna("")
        print(f"✅ Loaded {latest_inventory_file.name} ({len(self.xero_inventory)} rows)")

        self.dict_inventory = {}
        for _, row in self.xero_inventory.iterrows():
            code = str(row.get("*ItemCode", "")).strip()
            self.dict_inventory[code] = {
                "ItemName": str(row.get("ItemName", "")).strip(),
                "PurchasesAccount": str(row.get("PurchasesAccount", "")).strip(),
                "SalesAccount": str(row.get("SalesAccount", "")).strip()
            }

    # ----------------------
    # One-pass flatten (charges + packages)
    # ----------------------
    def _flatten_all_once(self):
        """Traverse composite ONCE and populate flat_charges & flat_packages."""
        charge_rows, pkg_rows = [], []
        c_append, p_append = charge_rows.append, pkg_rows.append

        for invoice in self.invoices.values():
            inv_num = getattr(invoice, "inv_num", "")
            inv_date = pd.to_datetime(getattr(invoice, "inv_date", None)) if getattr(invoice, "inv_date", None) else pd.NaT
            acct_num = getattr(invoice, "acct_num", "")

            # Invoice-level charges
            for ch in getattr(invoice, "inv_charge", {}).values():
                c_append({
                    "Invoice Number": inv_num,
                    "Invoice Date": inv_date,
                    "Account Number": acct_num,
                    "cust_id": "",
                    "Lead Shipment Number": "",
                    "Tracking Number": "",
                    "Charge_Cate_EN": getattr(ch, "charge_en", ""),
                    "Charge_Cate_CN": getattr(ch, "charge_cn", ""),
                    "ap_amt": float(getattr(ch, "ap_amt", 0) or 0),
                    "ar_amt": float(getattr(ch, "ar_amt", 0) or 0),
                })

            # Shipments (+ shipment charges + packages + package charges)
            for ship in getattr(invoice, "shipments", {}).values():
                cust_id = getattr(ship, "cust_id", "")
                main_trk = getattr(ship, "main_trk_num", "")

                # Shipment-level charges
                for ch in getattr(ship, "shipment_charge", {}).values():
                    c_append({
                        "Invoice Number": inv_num,
                        "Invoice Date": inv_date,
                        "Account Number": acct_num,
                        "cust_id": cust_id,
                        "Lead Shipment Number": main_trk,
                        "Tracking Number": main_trk, 
                        "Charge_Cate_EN": getattr(ch, "charge_en", ""),
                        "Charge_Cate_CN": getattr(ch, "charge_cn", ""),
                        "ap_amt": float(getattr(ch, "ap_amt", 0) or 0),
                        "ar_amt": float(getattr(ch, "ar_amt", 0) or 0),
                    })

                # Packages (and package-level charges)
                for pkg in getattr(ship, "packages", {}).values():
                    # package row for Packages sheet
                    w_lb = getattr(pkg, "billed_wgt", None)
                    try:
                        w_kg = round(float(w_lb) / 2.20462, 2) if w_lb not in (None, "", "nan") else None
                    except Exception:
                        w_kg = None

                    def to_cm(v):
                        try:
                            return round(float(v) * 2.54, 2)
                        except Exception:
                            return None

                    p_append({
                        "cust_id": cust_id,
                        "Invoice Number": inv_num,
                        "Invoice Date": inv_date,
                        "Lead Shipment Number": main_trk,
                        "Tracking Number": getattr(pkg, "trk_num", "") or main_trk,
                        "Billed Weight (kg)": w_kg,
                        "Length (cm)": to_cm(getattr(pkg, "length", "")),
                        "Width (cm)":  to_cm(getattr(pkg, "width", "")),
                        "Height (cm)": to_cm(getattr(pkg, "height", "")),
                        "Receiver Name":  getattr(ship.consignee, "contact", ""),
                        "Receiver Company":getattr(ship.consignee, "company", ""),
                        "Receiver City":  getattr(ship.consignee, "city", ""),
                        "Receiver State": getattr(ship.consignee, "state", ""),
                        "Receiver Postal":getattr(ship.consignee, "zipcode", ""),
                        "Sender Name":    getattr(ship.sender, "contact", ""),
                        "Sender City":    getattr(ship.sender, "city", ""),
                        "Sender State":   getattr(ship.sender, "state", ""),
                        "Sender Postal":  getattr(ship.sender, "zipcode", ""),
                    })

                    # Package-level charges
                    for ch in getattr(pkg, "charge_detail", {}).values():
                        c_append({
                            "Invoice Number": inv_num,
                            "Invoice Date": inv_date,
                            "Account Number": acct_num,
                            "cust_id": cust_id,
                            "Lead Shipment Number": main_trk,
                            "Tracking Number": getattr(pkg, "trk_num", "") or main_trk,
                            "Charge_Cate_EN": getattr(ch, "charge_en", ""),
                            "Charge_Cate_CN": getattr(ch, "charge_cn", ""),
                            "ap_amt": float(getattr(ch, "ap_amt", 0) or 0),
                            "ar_amt": float(getattr(ch, "ar_amt", 0) or 0),
                        })

        self.flat_charges = pd.DataFrame(charge_rows)
        self.flat_packages = pd.DataFrame(pkg_rows)

        if not self.flat_charges.empty:
            self.flat_charges["ap_amt"] = pd.to_numeric(self.flat_charges["ap_amt"], errors="coerce").fillna(0.0)
            self.flat_charges["ar_amt"] = pd.to_numeric(self.flat_charges["ar_amt"], errors="coerce").fillna(0.0)

        if not self.flat_packages.empty:
            for c in ["Billed Weight (kg)", "Length (cm)", "Width (cm)", "Height (cm)"]:
                if c in self.flat_packages.columns:
                    self.flat_packages[c] = pd.to_numeric(self.flat_packages[c], errors="coerce")

    def _ensure_flattened(self):
        if self.flat_charges is None or self.flat_charges.empty:
            self._flatten_all_once()

    # ----------------------
    # Cost splits
    # ----------------------
    def _split_costs(self):  # keep above for reuse by Xero
        self._ensure_flattened()
        df = self.flat_charges.copy()

        mask_general = df["Lead Shipment Number"].apply(is_blank)

        general_df = (
            df[mask_general]
            .groupby(["Charge_Cate_CN", "Charge_Cate_EN"], as_index=False)[["ap_amt", "ar_amt"]]
            .sum()
        )
        general_df["*AccountCode"] = general_df["Charge_Cate_EN"].map(self.GENERAL_ITEMCODE_MAP).fillna("7155")
        general_df["SourceType"] = "general"
        general_df["*ItemCode"] = ""

        cust_df = (
            df[~mask_general]
            .groupby("cust_id", as_index=False)[["ap_amt", "ar_amt"]]
            .sum()
        )
        for cid in SPECIAL_CUSTOMERS:
            mask = cust_df["cust_id"] == cid
            if mask.any(): 
                factor = self.dict_ar.get(cid, {}).get("Factor", 0.0)
                cust_df.loc[mask, "ar_amt"] = (
                    cust_df.loc[mask, "ap_amt"] * factor
                ).round(2)

        cust_df["*ItemCode"] = cust_df["cust_id"].astype(str).str[-4:]
        cust_df["*AccountCode"] = cust_df["*ItemCode"].map(
            lambda code: self.dict_inventory.get(code, {}).get("PurchasesAccount", "")
        )
        cust_df["SourceType"] = "customer"

        self.general_cost_df = general_df
        self.customer_cost_df = cust_df

    # ----------------------
    # Master export
    # ----------------------
    def export(self):
        """Export details + summaries to Excel (adds 'Summary for General Cost')."""
        self._ensure_flattened()
        self._split_costs()
        df = self.flat_charges.copy()

        summary_invoice = df.groupby("Invoice Number")[["ap_amt", "ar_amt"]].sum().reset_index()
        summary_customer = df.groupby("cust_id")[["ap_amt", "ar_amt"]].sum().reset_index()
        for cid in SPECIAL_CUSTOMERS:
            mask = summary_customer["cust_id"] == cid
            if mask.any():  # only update if cid exists
                factor = self.dict_ar.get(cid, {}).get("Factor", 0.0)
                summary_customer.loc[mask, "ar_amt"] = (
                    summary_customer.loc[mask, "ap_amt"] * factor
                ).round(2)

        output_file = self.output_path / "UPS_Invoice_Export.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Details", index=False)
            summary_invoice.to_excel(writer, sheet_name="Summary by Invoice", index=False)
            summary_customer.to_excel(writer, sheet_name="Summary by Customer", index=False)
            (self.general_cost_df[["Charge_Cate_CN", "ap_amt"]]
                .rename(columns={"Charge_Cate_CN": "费用类型（中文）", "ap_amt": "总应付金额"})
                .sort_values("总应付金额", ascending=False)
                .to_excel(writer, sheet_name="Summary for General Cost", index=False))
        print(f"📁 UPS invoice export saved to {output_file}")

    def _generate_special_customer_invoices(self):
        tried_encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        header_path = Path(__file__).resolve().parent / "data" / "mappings" / "OriHeadr.csv"
        headers = pd.read_csv(header_path)
        header_list = headers["Column Name"].tolist()
        
        # build once from flat_packages for speed
        if self.flat_packages.empty:
            print("⚠️ No packages to derive keys from.")
            return

        for cid, meta in SPECIAL_CUSTOMERS.items():
            acct_codes = set(meta.get("accounts", []))

            # get lead/trk sets for THIS cid from flat_packages
            sub = self.flat_packages[self.flat_packages["cust_id"] == cid]
            lead_set = set(sub["Lead Shipment Number"].dropna().astype(str).unique())
            trk_set  = set(sub["Tracking Number"].dropna().astype(str).unique())

            dfs = []

            for file in self.raw_path.glob("*.csv"):
                fname = file.name

                # Case 1: whole file belongs to this special by acct code in filename
                has_acct = any(ac in fname for ac in acct_codes) or any(
                    len(fname) >= 21 and fname[15:21] == ac for ac in acct_codes
                )

                # Read raw with no headers for both branches to keep shape consistent
                df = None
                for enc in tried_encs:
                    try:
                        df = pd.read_csv(file, header=None, dtype=str, encoding=enc, low_memory=False)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"⚠️ Could not read {fname} with {enc}: {e}")
                        break
                if df is None:
                    print(f"❌ Skipped {fname}: unable to decode with {tried_encs}")
                    continue

                if has_acct:
                    dfs.append(df)
                    continue

                # Case 2: general invoices — select only rows whose col14/col21 match our keys
                if df.shape[1] >= 21:
                    lead_col, trk_col = 13, 20
                    df.iloc[:, lead_col] = df.iloc[:, lead_col].astype(str).str.strip()
                    df.iloc[:, trk_col]  = df.iloc[:, trk_col].astype(str).str.strip()
                    mask = df.iloc[:, lead_col].isin(lead_set) | df.iloc[:, trk_col].isin(trk_set)
                    sub_rows = df.loc[mask]
                    if not sub_rows.empty:
                        dfs.append(sub_rows)

            if not dfs:
                # nothing matched for this special — skip quietly
                continue

            combined_inv = pd.concat(dfs, ignore_index=True)
            output_path = self.output_path / f"RAW_CustomerInvoice_{cid}.csv"
            combined_inv.to_csv(output_path, header=header_list, index=False)
            print(f"📁 Customer invoice exported: {output_path}")

    def _generate_general_customer_invoices(self):
        ar_lines_all = self.flat_charges.copy()
        ar_lines_all = ar_lines_all[(~ar_lines_all["Lead Shipment Number"].apply(is_blank))]
        ar_lines_all = ar_lines_all[ar_lines_all["ar_amt"] != 0]

        pkg_all = self.flat_packages.copy()

        cust_ids = sorted(
            set(ar_lines_all.get("cust_id", pd.Series(dtype=str)).dropna().unique())
            | set(pkg_all.get("cust_id", pd.Series(dtype=str)).dropna().unique())
        )
        
        # Exclude special customers
        specials = SPECIAL_CUSTS
        normal_custs = [c for c in cust_ids if c not in specials]

        for cid in normal_custs:
            ar_sub = ar_lines_all[ar_lines_all["cust_id"] == cid].copy()
            pkg_sub = pkg_all[pkg_all["cust_id"] == cid].copy() if not pkg_all.empty else pd.DataFrame()

            # AR Summary
            if not ar_sub.empty:
                ar_summary = (
                    ar_sub.groupby("Charge_Cate_CN", as_index=False)["ar_amt"]
                    .sum()
                    .sort_values("ar_amt", ascending=False)
                    .rename(columns={"Charge_Cate_CN": "费用类型（中文）", "ar_amt": "金额"})
                )
                grand_total = round(float(ar_sub["ar_amt"].sum()), 2)
            else:
                ar_summary = pd.DataFrame(columns=["费用类型（中文）", "金额"])
                grand_total = 0.00

            # Cover
            cover = pd.DataFrame({
                "Field": ["Customer ID", "Batch Num", "Charge Total"],
                "Value": [cid, self.batch_number, grand_total],
            })

            # AR Pivot (detail)
            # 1. Pivot table
            pivot_df = ar_sub.pivot_table(
                index="Tracking Number",
                columns="Charge_Cate_CN",
                values="ar_amt",
                aggfunc="sum",
                fill_value=0
            )
            # 2. Ensure "运费"(transporation fee) is the first column
            cols = pivot_df.columns.tolist()
            if "运费" in cols:
                cols.remove("运费")
                cols = ["运费"] + cols
            pivot_df = pivot_df[cols]
            # 3. Optional: reset index for export
            pivot_df = pivot_df.reset_index()

            # Write
            out = self.output_path / f"CustomerInvoice_{cid}.xlsx"
            with pd.ExcelWriter(out, engine="xlsxwriter") as w:
                cover.to_excel(w, sheet_name="Invoice", index=False)
                ar_summary.to_excel(w, sheet_name="AR Summary", index=False)
                pivot_df.to_excel(w, sheet_name="Charge Pivot", index=False)
                (pkg_sub if not pkg_sub.empty else pd.DataFrame(columns=["（no packages）"])).to_excel(
                    w, sheet_name="Packages", index=False
                )
            print(f"📁 Customer invoice exported: {out}")

    # ----------------------
    # Customer invoices (one Excel per customer)
    # ----------------------
    def generate_customer_invoices(self):
        """
        Create one Excel per customer with:
          - Invoice (cover: Customer, Batch, AR Total)
          - AR Summary (by Charge_Cate_CN)
          - AR Lines   (detail lines, AR only)
          - Packages   (per-package info)
        2 groups of customers:
          - Special customer: total ar amount = total ap amount * factor
          - General customer: total ar amount = sum(ar charges for each item)
        """
        self._ensure_flattened()
        self._generate_special_customer_invoices()
        self._generate_general_customer_invoices()
        
    def _build_special_keys(self, matched_df: pd.DataFrame, cid: str):
        '''
        Build key sets from matched_df
        Return lead shipment tracking number and pkg tracking number for special customers
        '''
        if  cid in SPECIAL_CUSTOMERS:
            sub = matched_df[matched_df["cust_id"] == cid]
            lead_nums = set(sub["Lead Shipment Number"].dropna().astype(str).unique())
            trk_nums  = set(sub["Tracking Number"].dropna().astype(str).unique())

        return lead_nums, trk_nums    

    # ----------------------
    # YDD AP Template
    # ----------------------
    def generate_ydd_ap_template(self):
        self._ensure_flattened()

        # Exclude special customers
        df = self.flat_charges.copy()
        df = df[~df["cust_id"].isin(SPECIAL_CUSTS)]
        df = df[~df["Lead Shipment Number"].apply(is_blank)]

        df["ap_amt"] = pd.to_numeric(df["ap_amt"], errors="coerce").fillna(0)

        # ✅ Get billed weight per shipment from flat_packages (NOT flat_charges)
        if not self.flat_packages.empty and "Billed Weight (kg)" in self.flat_packages.columns:
            bw_per_ship = (
                self.flat_packages
                .dropna(subset=["Lead Shipment Number"])
                .groupby("Lead Shipment Number")["Billed Weight (kg)"]
                # If you want total weight across packages, use .sum() here
                .max()                 # ← commonly billed weight per shipment is the max
                .round(2)
            )
        else:
            bw_per_ship = pd.Series(dtype=float)

        # Group AP by customer + shipment + charge
        grouped = (
            df.groupby(["cust_id", "Lead Shipment Number", "Charge_Cate_CN"], as_index=False)
            .agg({"ap_amt": "sum"})
        )

        # ✅ Map billed weight into the grouped rows
        grouped["代理计费重"] = grouped["Lead Shipment Number"].map(bw_per_ship).fillna("")

        # Rename + order columns for YDD
        grouped = grouped.rename(columns={
            "cust_id": "客户编号",
            "Lead Shipment Number": "转单号",
            "Charge_Cate_CN": "费用名称",
            "ap_amt": "金额",
        })
        grouped["金额"] = grouped["金额"].round(2)
        grouped = grouped[["客户编号", "转单号", "费用名称", "金额", "代理计费重"]]

        output_file = self.output_path / "YDD_AP_Template.xlsx"
        grouped.to_excel(output_file, index=False)
        print(f"📁 YiDiDa AP template saved to {output_file}")

    # ----------------------
    # YDD AR Template
    # ----------------------
    def generate_ydd_ar_template(self):
        """Generate YiDiDa AR template and export as Excel file."""
        self._ensure_flattened()

        # Exclude special customers
        df = self.flat_charges.copy()
        df = df[~df["cust_id"].isin(SPECIAL_CUSTS)]
        df = df[~df["Lead Shipment Number"].apply(is_blank)]

        df["ar_amt"] = pd.to_numeric(df["ar_amt"], errors="coerce").fillna(0)

        grouped = (
            df.groupby(["Lead Shipment Number", "Charge_Cate_CN", "cust_id"], as_index=False)
            .agg({"ar_amt": "sum"})
        )
        grouped["ar_amt"] = grouped["ar_amt"].round(2)

        ar_df = pd.DataFrame({
            "主提单号/客户单号/系统单号": grouped["Lead Shipment Number"],
            "子转单号/子系统单号": "",
            "费用名": grouped["Charge_Cate_CN"],
            "金额": grouped["ar_amt"],
            "币种": "USD",
            "结算单位代码": grouped["cust_id"],
            "内部备注": "",
            "公开备注": "",
            "计量单位": "",
            "覆盖追加策略": "追加",
            "自动对账": "Y"
        })

        output_file = self.output_path / "YDD_AR_Template.xlsx"
        ar_df.to_excel(output_file, index=False)
        print(f"📁 YiDiDa AR template saved to {output_file}")

    # ----------------------
    # Xero AP Template
    # ----------------------
    def generate_xero_ap_template(self):
        self._split_costs()
        combined_df = pd.concat([self.general_cost_df, self.customer_cost_df], ignore_index=True, sort=False)

        combined_df["*ContactName"] = "UPS"
        combined_df["*InvoiceNumber"] = self.batch_number
        combined_df["*InvoiceDate"] = self.inv_date
        combined_df["*DueDate"] = self.inv_date + timedelta(days=30)
        combined_df["InventoryItemCode"] = combined_df["*ItemCode"]

        combined_df.loc[combined_df["SourceType"] == "general", "Description"] = combined_df["Charge_Cate_EN"]
        combined_df.loc[combined_df["SourceType"] == "customer", "Description"] = combined_df["*ItemCode"].map(
            lambda code: self.dict_inventory.get(code, {}).get("ItemName", "UPS Services")
        )

        combined_df["*Quantity"] = 1
        combined_df["*UnitAmount"] = combined_df["ap_amt"]
        combined_df["*TaxType"] = "Tax Exempt"

        final_cols = [
            "*ContactName", "*InvoiceNumber", "*InvoiceDate", "*DueDate",
            "InventoryItemCode", "Description", "*Quantity", "*UnitAmount",
            "*AccountCode", "*TaxType"
        ]
        output_file = self.output_path / "Xero_AP_Template.csv"
        combined_df[final_cols].to_csv(output_file, index=False)
        print(f"📁 Xero AP template saved to {output_file}")

    # ----------------------
    # Xero AR Template
    # ----------------------
    def generate_xero_ar_template(self):
        if not hasattr(self, "customer_cost_df") or self.customer_cost_df.empty:
            raise ValueError("Run generate_xero_ap_template() first.")

        df = self.customer_cost_df.copy()

        df["*ContactName"] = df["cust_id"].map(
            lambda cid: self.dict_contacts.get(cid, {}).get("ContactName", "")
        )
        today = pd.Timestamp.today().normalize()
        df["*InvoiceNumber"] = df["cust_id"].astype(str) + "-" + self.batch_number
        df["*InvoiceDate"] = today
        df["*DueDate"] = today + timedelta(days=30)
        df["InventoryItemCode"] = df["*ItemCode"]
        df["*Description"] = "UPS Services"
        df["*Quantity"] = 1
        df["*UnitAmount"] = df["ar_amt"]
        df["*AccountCode"] = df["*ItemCode"].map(
            lambda code: self.dict_inventory.get(code, {}).get("SalesAccount", "")
        )
        df["*TaxType"] = "Tax Exempt"

        final_cols = [
            "*ContactName", "*InvoiceNumber", "*InvoiceDate", "*DueDate",
            "InventoryItemCode", "*Description", "*Quantity", "*UnitAmount",
            "*AccountCode", "*TaxType"
        ]
        output_file = self.output_path / "Xero_AR_Template.csv"
        df[final_cols].to_csv(output_file, index=False)
        print(f"📁 Xero AR template saved to {output_file}")

    # ----------------------
    # Combined runner
    # ----------------------
    def generate_xero_templates(self):
        self.generate_xero_ap_template()
        self.generate_xero_ar_template()