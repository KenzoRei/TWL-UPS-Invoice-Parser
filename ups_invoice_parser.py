import shutil
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from datetime import timedelta

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

class UpsInvLoader:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.invoices = []  # Placeholder for loaded invoice files
        self.basic_info = []  # List of dicts like {'InvoiceNumber':..., 'Date':...}

    def validate_format(self) -> bool:
        """
        Validate the format of each file in the input directory.
        Return True if all pass, False otherwise.
        """
        # TODO: implement format checking logic
        print("Validating invoice formats...")
        return True

    def extract_basic_info(self):
        """
        Extract invoice number, date, or other identifiers.
        Stores results in self.basic_info.
        """
        # TODO: parse metadata from invoices
        print("Extracting basic invoice information...")
        self.basic_info = [
            {"filename": file.name, "InvoiceNumber": "123456", "Date": "2024-06-01"}
            for file in self.input_dir.glob("*.xlsx")
        ]

    def archive_raw_invoices(self, output_dir: str) -> Path:
        """
        Save raw invoice files to a specified directory.
        Returns the path to the archived folder.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for file in self.input_dir.glob("*.xlsx"):
            shutil.copy(file, output_path / file.name)
        
        print(f"Archived raw invoices to {output_path}")
        return output_path
    
# How to use:
# loader = UpsInvLoader("/path/to/loading_folder")
# if loader.validate_format():
#     loader.extract_basic_info()
#     output_dir = loader.archive_raw_invoices("/path/to/archive_folder")


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

    def load_invoices(self):
        """Load CSV invoice files with correct dtypes and dates."""
        self.raw_dataframes = []
        for file in self.file_list:
            df = pd.read_csv(
                file, 
                header=None,
                names=self.all_col_name,
                dtype=self.dtype_map
                )
            df = df.loc[:, self.filtered_col_name]
            for date_col in self.date_cols:
                df[date_col] = pd.to_datetime(df[date_col])
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

    def get_normalized_data(self) -> pd.DataFrame:
        return self.normalized_df

# How to use:
# from pathlib import Path
# from ups_invoice_parser import UpsInvNormalizer

# # Step 1: Gather all invoice files
# base_path = Path(__file__).resolve().parent
# invoice_folder = base_path / "data/raw_invoices"
# output_path = base_path / "output/normalized_output.xlsx"
# file_list = list(invoice_folder.glob("*.csv"))

# # Step 2: Create the normalizer instance
# normalizer = UpsInvNormalizer(file_list)


# # Step 3: Run normalization steps
# normalizer.load_invoices()
# normalizer.merge_invoices()
# normalizer.standardize_invoices()

# # Step 4: Get and inspect the result
# df = normalizer.get_normalized_data()
# print(df.head())  # Print top 5 rows
# print(df.columns.tolist())  # Print all column names
# df.to_excel(output_path, index=False)



# Match normalized UPS invoice with customer ID   

import math 
class UpsCustomerMatcher:
    def __init__(self, normalized_df: pd.DataFrame):
        """
        :param normalized_df: DataFrame output from normalizer.get_normalized_data()
        """
        self.df = normalized_df.copy()  # Keep original safe
        if "cust_id" not in self.df.columns: # Add empty column if not exist
            self.df["cust_id"] = ""
        self.df["Charge_Cate_EN"] = ""
        self.df["Charge_Cate_CN"] = ""
        self.DEFAULT_CUST_ID = "F000999"
        self.base_path = Path(__file__).resolve().parent
        self.mapping_cust = self.base_path / "input" / "数据列表.xlsx"               # for cust.id mapping
        self.mapping_cust_df = pd.DataFrame()
        self.mapping_pickup = self.base_path / "data" / "mappings" / "Pickups.csv"  # for daily pickup mapping
        self.mapping_pickup_df = pd.DataFrame()
        self.mapping_chrg = self.base_path / "data" / "mappings" / "Charges.csv"    # for charge category mapping
        self.mapping_chrg_df = pd.DataFrame()
        self.mapping_ar = self.base_path / "data" / "mappings" / "ARCalculator.csv" # for ar amount amplifier & modifier flag mapping
        self.mapping_ar_df = pd.DataFrame() 
        self.trk_to_cust = {}

        self._load_mapping()        

    def _load_mapping(self):

        # Load 数据列表.xlsx into self.mapping_cust_df. and flatten it into dict self.trk_to_cust
        self.mapping_cust_df = pd.read_excel(self.mapping_cust)
        required_cols = {"子转单号", "客户编号", "转单号"}
        if not required_cols.issubset(set(self.mapping_cust_df.columns)):
            raise ValueError(f"Mapping file missing required columns: {required_cols}")
        self._build_lookup_dict()
        
        # Load Charges.csv into dict self.dict_chrg     
        self.mapping_chrg_df = pd.read_csv(self.mapping_chrg)
        self.dict_chrg = self.mapping_chrg_df.set_index(self.mapping_chrg_df.columns[0]).to_dict(orient="index")

        # Load Pickups.csv into dict self.dict_pickup
        self.mapping_pickup_df = pd.read_csv(self.mapping_pickup)
        self.dict_pickup = self.mapping_pickup_df.set_index(self.mapping_pickup_df.columns[0]).to_dict(orient="index")

        # Load ARCalculator.csv into dict self.dict_ar
        self.mapping_ar_df = pd.read_csv(self.mapping_ar)
        self.dict_ar = self.mapping_ar_df.set_index(self.mapping_ar_df.columns[0]).to_dict(orient="index")

    def _build_lookup_dict(self):
        """Flatten mapping and build dictionary: {Tracking Number: (客户编号, 转单号)}."""
        mapping_records = []

        for _, row in self.mapping_cust_df.iterrows():
            sub_trks = str(row["子转单号"]).split(",")
            for trk in sub_trks:
                cleaned_trk = trk.replace(" ", "").replace("[", "").replace("]", "")
                if cleaned_trk:
                    mapping_records.append({
                        "Tracking Number": cleaned_trk,
                        "客户编号": row["客户编号"],
                        "转单号": row["转单号"]
                    })

        flat_df = pd.DataFrame(mapping_records)
        self.trk_to_cust = {
            row["Tracking Number"]: (row["客户编号"], row["转单号"])
            for _, row in flat_df.iterrows()
        }

    def match_customers(self):
        """
        1. Match and overwrite cust_id and Lead Shipment Number in self.df
        2. Generate YiDiDa import template for unmatched charges
        3. Calculate AR amount
        """
        exception_rows = []
        for idx, row in self.df.iterrows():
            cust_id, lead_shipment = np.nan, np.nan

            # classify charges
            category_en, category_cn = self._charge_classifier(row)
            self.df.at[idx, "Charge_Cate_EN"] = category_en
            self.df.at[idx, "Charge_Cate_CN"] = category_cn

            # matchup cust.id
            trk_num = str(row["Tracking Number"])
            if trk_num in self.trk_to_cust:
                cust_id, lead_shipment = self.trk_to_cust[trk_num]
            else:
                exception_rows.append(row.to_dict())
                cust_id = self._exception_handler(row)
                if is_blank(row["Lead Shipment Number"]):
                    lead_shipment = trk_num
                else:
                    lead_shipment = row["Lead Shipment Number"]
                # other possible handlings
            self.df.at[idx, "cust_id"] = cust_id
            # REMINDER: overwrite lead shipment# and trk# may lead to info loss!!
            self.df.at[idx, "Lead Shipment Number"] = lead_shipment
            if is_blank(row["Tracking Number"]):
                self.df.at[idx, "Tracking Number"] = lead_shipment
 
        self._ar_calculator()
        df_exception = pd.DataFrame(exception_rows)
        self._template_generator(df_exception)

        # Save all rows with undefined charge categories
        unmapped_charges = self.df[self.df["Charge_Cate_EN"] == "Not Defined Charge"]
        if not unmapped_charges.empty:
            output_path = self.base_path / "output" / "UnmappedCharges.xlsx"
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            unmapped_charges.to_excel(output_path, index=False)

    def _exception_handler(self, row: pd.Series) -> str:
        """Manually match a shipment with cust_id and return it."""

        # 0. pass cust_id if it already exists
        if not is_blank(row["cust_id"]):
            return row["cust_id"]
        
        # 1. Import department pickup
        elif (
            row["Lead Shipment Number"].startswith("2") and
            "import" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000223"

        # 2. UPEX (vermilion match)
        elif (
            "vermilion" in row["Sender Name"].lower() or
            "vermilion" in row["Sender Company Name"].lower() or
            "vermilion" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000215"

        # 3. Bondex
        elif row["Account Number"] in ["K5811C", "F03A44"]:
            return "F000281"

        # 4. SPT
        elif row["Account Number"] == "HE6132":
            return "F000208"

        # 5. Transware
        elif not is_blank(row["Tracking Number"]) or not is_blank(row["Lead Shipment Number"]):
            return "F000222"

        # 6. Daily Pickup logic
        elif row["Charge_Cate_EN"] in ["Daily Pickup", "Daily Pickup - Fuel"]:
            return self.dict_pickup.get(row["Account Number"], {}).get("Cust.ID", self.DEFAULT_CUST_ID)

        # 7. Generic cost rules
        elif row["Charge_Cate_EN"] in ["processing fee", "SCC audit fee", "POD fee"]:
            return self.DEFAULT_CUST_ID

        # 8. Fallback
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
        """Generate an YiDiDa template and save is to output folder"""
        df_exceptions = df
        # waiting edit
        
        
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
    
# how to use:
# from ups_invoice_parser import UpsInvNormalizer  # Your existing normalizer class
# from ups_customer_matcher import UpsCustomerMatcher  # Assuming you put it in new file

# # Get normalized data
# normalizer = UpsInvNormalizer(file_list)
# normalizer.load_invoices()
# normalizer.merge_invoices()
# normalizer.standardize_invoices()
# normalized_df = normalizer.get_normalized_data()

# # Create matcher
# matcher = UpsCustomerMatcher(normalized_df)
# matcher.match_customers()
# matched_df = matcher.get_matched_data()

# # Save or inspect
# matched_df.to_excel("matched_output.xlsx", index=False)
# print("✅ Matching completed and saved.")

from datetime import datetime
from models import Invoice, Shipment, Package, Charge, Location
import logging
import pickle
import os

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

            # for general invoice cost (don't allocate to costomer)
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

        print(f"✅ Invoices saved to {file_path}")

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
        # Store composite invoices
        self.invoices = invoices
        self.flatten_inv = pd.DataFrame()

        # Setup output path for batch folder
        self.output_path = Path(__file__).resolve().parent / "output"
        first_invoice = next(iter(self.invoices.values()))
        self.batch_number = getattr(first_invoice, "batch_num", "unknown_batch")
        self.inv_date = pd.to_datetime(first_invoice.inv_date)  # ensure datetime
        self.output_path = self.output_path / self.batch_number
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Load mappings
        self._load_mappings()

    # ----------------------
    # Mapping loader
    # ----------------------
    def _load_mappings(self):
        """Load Xero Contacts, InventoryItems, and ARCalculator mapping as DataFrames and dicts."""
        mappings_path = Path(__file__).resolve().parent / "data" / "mappings"

        # === Contacts.csv ===
        contacts_file = mappings_path / "Contacts.csv"
        if not contacts_file.exists():
            raise FileNotFoundError(f"❌ Contacts.csv not found in {mappings_path}")
        self.xero_contacts = pd.read_csv(contacts_file, dtype=str).fillna("")
        print(f"✅ Loaded Contacts.csv ({len(self.xero_contacts)} rows)")

        # Build dict keyed by AccountNumber
        self.dict_contacts = {}
        for _, row in self.xero_contacts.iterrows():
            acct = row["AccountNumber"].strip()
            contact_name_col = "*ContactName" if "*ContactName" in row else "ContactName"
            self.dict_contacts[acct] = {
                "ContactName": row[contact_name_col].strip(),
                "EmailAddress": row["EmailAddress"].strip(),
                "POAddress": {
                    "AttentionTo": row["POAttentionTo"].strip(),
                    "Line1": row["POAddressLine1"].strip(),
                    "Line2": row["POAddressLine2"].strip(),
                    "Line3": row["POAddressLine3"].strip(),
                    "Line4": row["POAddressLine4"].strip(),
                    "City": row["POCity"].strip(),
                    "Region": row["PORegion"].strip(),
                    "ZipCode": row["POZipCode"].strip(),
                    "Country": row["POCountry"].strip()
                },
                "SAAddress": {
                    "AttentionTo": row["SAAttentionTo"].strip(),
                    "Line1": row["SAAddressLine1"].strip(),
                    "Line2": row["SAAddressLine2"].strip(),
                    "Line3": row["SAAddressLine3"].strip(),
                    "Line4": row["SAAddressLine4"].strip(),
                    "City": row["SACity"].strip(),
                    "Region": row["SARegion"].strip(),
                    "ZipCode": row["SAZipCode"].strip(),
                    "Country": row["SACountry"].strip()
                },
                "PhoneNumber": row["PhoneNumber"].strip(),
                "MobileNumber": row["MobileNumber"].strip()
            }

        # === InventoryItems-YYYYMMDD.csv ===
        inventory_files = sorted(mappings_path.glob("InventoryItems-*.csv"), reverse=True)
        if not inventory_files:
            raise FileNotFoundError(f"❌ No InventoryItems-*.csv found in {mappings_path}")
        latest_inventory_file = inventory_files[0]
        self.xero_inventory = pd.read_csv(latest_inventory_file, dtype=str).fillna("")
        print(f"✅ Loaded {latest_inventory_file.name} ({len(self.xero_inventory)} rows)")

        self.dict_inventory = {}
        for _, row in self.xero_inventory.iterrows():
            code = row["*ItemCode"].strip()
            self.dict_inventory[code] = {
                "ItemName": row["ItemName"].strip(),
                "PurchasesAccount": row["PurchasesAccount"].strip(),
                "SalesAccount": row["SalesAccount"].strip()
            }
        
    # ----------------------
    # Flatten + ensure numeric
    # ----------------------
    def flatten(self) -> pd.DataFrame:
        rows = []
        for inv_num, invoice in self.invoices.items():
            for cate, charge in invoice.inv_charge.items():
                rows.append(self._build_row(invoice, None, None, cate, charge))
            for ship_num, shipment in invoice.shipments.items():
                for cate, charge in shipment.shipment_charge.items():
                    rows.append(self._build_row(invoice, shipment, None, cate, charge))
                for pkg_num, package in shipment.packages.items():
                    for cate, charge in package.charge_detail.items():
                        rows.append(self._build_row(invoice, shipment, package, cate, charge))
        return pd.DataFrame(rows)

    def _build_row(self, invoice, shipment, package, cate, charge):
        return {
            "Invoice Number": invoice.inv_num,
            "Account Number": invoice.acct_num,
            "Lead Shipment Number": shipment.main_trk_num if shipment else "",
            "Billed Weight": package.billed_wgt if package else "",
            "Charge_Cate_EN": charge.charge_en,
            "Charge_Cate_CN": charge.charge_cn,
            "ap_amt": round(charge.ap_amt, 2),
            "ar_amt": round(charge.ar_amt, 2),
            "cust_id": shipment.cust_id if shipment else ""
        }

    def _ensure_flattened(self):
        if self.flatten_inv.empty:
            self.flatten_inv = self.flatten()
        self.flatten_inv["ap_amt"] = pd.to_numeric(self.flatten_inv["ap_amt"], errors="coerce").fillna(0)
        self.flatten_inv["ar_amt"] = pd.to_numeric(self.flatten_inv["ar_amt"], errors="coerce").fillna(0)

    # ----------------------
    # Split cost breakdown
    # ----------------------
    def _split_costs(self):
        self._ensure_flattened()
        df = self.flatten_inv.copy()

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
        cust_df["*ItemCode"] = cust_df["cust_id"].astype(str).str[-4:]
        cust_df["*AccountCode"] = cust_df["*ItemCode"].map(
            lambda code: self.dict_inventory.get(code, {}).get("PurchasesAccount", "")
        )
        cust_df["SourceType"] = "customer"

        self.general_cost_df = general_df
        self.customer_cost_df = cust_df

    def export(self):
        """Export full invoice details and summaries to Excel."""
        self._ensure_flattened()
        df = self.flatten_inv.copy()

        summary_invoice = df.groupby("Invoice Number")[["ap_amt", "ar_amt"]].sum().reset_index()
        summary_customer = df.groupby("cust_id")[["ap_amt", "ar_amt"]].sum().reset_index()

        output_file = self.output_path / "UPS_Invoice_Export.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Details", index=False)
            summary_invoice.to_excel(writer, sheet_name="Summary by Invoice", index=False)
            summary_customer.to_excel(writer, sheet_name="Summary by Customer", index=False)

        print(f"✅ UPS invoice export saved to {output_file}")
    
    # ----------------------
    # YDD AP Template
    # ----------------------
    def generate_ydd_ap_template(self):
        """Generate YiDiDa AP template and export as Excel file."""
        self._ensure_flattened()
        df = self.flatten_inv[~self.flatten_inv["Lead Shipment Number"].apply(is_blank)].copy()

        # Ensure numeric AP amount
        df["ap_amt"] = pd.to_numeric(df["ap_amt"], errors="coerce").fillna(0)

        # Get billed weight in KG per shipment
        billed_wgt_kg = (
            pd.to_numeric(df["Billed Weight"], errors="coerce")
            .groupby(df["Lead Shipment Number"])
            .max()
            .apply(lambda w: round(w / 2.204, 2) if pd.notna(w) else "")
        )

        # Group AP by customer, shipment, and charge category
        grouped = (
            df.groupby(["cust_id", "Lead Shipment Number", "Charge_Cate_CN"], as_index=False)
            .agg({"ap_amt": "sum"})
        )

        # Add billed weight
        grouped["代理计费重"] = grouped["Lead Shipment Number"].map(billed_wgt_kg)

        # Rename to match template
        grouped = grouped.rename(columns={
            "cust_id": "客户编号",
            "Lead Shipment Number": "转单号",
            "Charge_Cate_CN": "费用名称",
            "ap_amt": "金额"
        })

        # Round 金额
        grouped["金额"] = grouped["金额"].round(2)

        # Final column order
        grouped = grouped[["客户编号", "转单号", "费用名称", "金额", "代理计费重"]]

        # Save to Excel
        output_file = self.output_path / "YDD_AP_Template.xlsx"
        grouped.to_excel(output_file, index=False)
        print(f"✅ YiDiDa AP template saved to {output_file}")

    # ----------------------
    # YDD AR Template
    # ----------------------
    def generate_ydd_ar_template(self):
        """Generate YiDiDa AR template and export as Excel file."""
        self._ensure_flattened()
        df = self.flatten_inv[~self.flatten_inv["Lead Shipment Number"].apply(is_blank)].copy()

        # Ensure numeric AP amount
        df["ap_amt"] = pd.to_numeric(df["ap_amt"], errors="coerce").fillna(0)

        # Group AP by Lead Shipment Number + Charge Category
        grouped = (
            df.groupby(["Lead Shipment Number", "Charge_Cate_CN", "cust_id"], as_index=False)
            .agg({"ap_amt": "sum"})
        )

        # Round amount
        grouped["ap_amt"] = grouped["ap_amt"].round(2)

        # Build template DataFrame
        ar_df = pd.DataFrame({
            "主提单号/客户单号/系统单号": grouped["Lead Shipment Number"],
            "子转单号/子系统单号": "",  # blank
            "费用名": grouped["Charge_Cate_CN"],
            "金额": grouped["ap_amt"],
            "币种": "USD",
            "结算单位代码": grouped["cust_id"],
            "内部备注": "",
            "公开备注": "",
            "计量单位": "",
            "覆盖追加策略": "追加",
            "自动对账": "Y"
        })

        # Save to Excel
        output_file = self.output_path / "YDD_AR_Template.xlsx"
        ar_df.to_excel(output_file, index=False)
        print(f"✅ YiDiDa AR template saved to {output_file}")


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
        print(f"✅ Xero AP template saved to {output_file}")

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
        df["*InvoiceNumber"] = df["cust_id"] + "-" + self.batch_number
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
        print(f"✅ Xero AR template saved to {output_file}")

    # ----------------------
    # Combined runner
    # ----------------------
    def generate_xero_templates(self):
        self.generate_xero_ap_template()
        self.generate_xero_ar_template()