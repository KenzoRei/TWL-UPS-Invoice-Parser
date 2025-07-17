import os
import shutil
from pathlib import Path
from typing import List
import pandas as pd
import re

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
                if row["Lead Shipment Number"] == '':
                    lead_shipment = trk_num
                # other possible handlings
            self.df.at[idx, "cust_id"] = cust_id
            self.df.at[idx, "Lead Shipment Number"] = lead_shipment

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

        # 1. Import department pickup
        if (
            row["Lead Shipment Number"].startswith("2") and
            "import" in row["Shipment Reference Number 1"].lower()
        ):
            return self.DEFAULT_CUST_ID

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
        elif row["Tracking Number"] != "nan" or row["Lead Shipment Number"] != "nan":
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
        
        if row["Tracking Number"] == "nan" and row["Charge Category Detail Code"] == "SVCH":
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
        empty_cust_id_rows = self.df[self.df["cust_id"] == "nan"]
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
            (self.df["Flag_Modifier"] == "")
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


from models import Invoice, Shipment, Package, Charge, Location
class UpsInvoiceBuilder:
    def __init__(self, normalized_df: pd.DataFrame):
        self.df = normalized_df
        self.invoices: dict[str, Invoice] = {}

    def build_invoices(self):
        """Convert normalized DataFrame into nested Invoice → Shipment → Package → Charge structure."""
        # Your grouping & object creation logic here

    def get_invoices(self) -> dict[str, Invoice]:
        """Return all constructed Invoice objects."""
        return self.invoices
    
