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
        df = self.normalized_df

        # Convert to datetime after columns are renamed
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

        df['Account Number'] = df['Account Number'].astype(str).str[-6:]
        df['Invoice Number'] = df['Invoice Number'].astype(str).str[-9:]

        df['Package Dimensions'] = df['Package Dimensions'].str.replace(' ', '', regex=False)
        dims = df['Package Dimensions'].str.split('x', expand=True).astype(float)
        dims.columns = ['Billed Length', 'Billed Width', 'Billed Height']
        idx = df.columns.get_loc('Package Dimensions') + 1
        for i, col in enumerate(dims.columns):
            df.insert(idx + i, col, dims[col])
        
        df['Place Holder 35'] = df['Place Holder 35'].str.replace(' ', '', regex=False)
        dims2 = df['Place Holder 35'].str.split('x', expand=True).astype(float)
        dims2.columns = ['Entered Length', 'Entered Width', 'Entered Height']
        idx2 = df.columns.get_loc('Place Holder 35') + 1
        for i, col in enumerate(dims2.columns):
            df.insert(idx2 + i, col, dims2[col])

        insert_idx = df.columns.get_loc('Incentive Amount')
        df.insert(insert_idx, 'Basis Amount', df['Incentive Amount'] + df['Net Amount'])

        charge_idx = df.columns.get_loc('Charge Description') + 1
        df.insert(charge_idx, 'Category_EN', '')
        df.insert(charge_idx + 1, 'Category_CN', '')

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
        self.df["cust_id"] = ""  # Add empty column if not exist
        self.base_path = Path(__file__).resolve().parent
        self.mapping_path = self.base_path / "input" / "数据列表.xlsx"
        self.mapping_df = pd.DataFrame()
        self.trk_to_cust = {}

        self._load_mapping()
        self._build_lookup_dict()

    def _load_mapping(self):
        """Load 数据列表.xlsx into self.mapping_df."""
        self.mapping_df = pd.read_excel(self.mapping_path)
        required_cols = {"子转单号", "客户编号", "转单号"}
        if not required_cols.issubset(set(self.mapping_df.columns)):
            raise ValueError(f"Mapping file missing required columns: {required_cols}")

    def _build_lookup_dict(self):
        """Flatten mapping and build dictionary: {Tracking Number: (客户编号, 转单号)}."""
        mapping_records = []

        for _, row in self.mapping_df.iterrows():
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
        """Match and overwrite cust_id and Lead Shipment Number in self.df."""
        col_name = ["客户代码", "转单号", "承运商子单号", "客户单号(文本格式)", "收货渠道", "目的地国家", 
                    "件数", "总实重", "长", "宽", "高", "收件人姓名", "收件公司", "收件人地址一", "收件人地址二", 
                    "城市", "省份/洲名简码", "邮编", "收件人电话", "包裹类型", "报关方式", "付税金", "中文品名", 
                    "英文品名", "海关编码", "数量", "币种", "单价", "总价", "购买保险", "寄件人姓名", "寄件公司", 
                    "寄件人地址一", "寄件人地址二", "寄件人城市", "寄件人州名", "寄件人邮编", "寄件人国家", "寄件电话", 
                    "货物特性", "收货备注", "预报备注", "idx"
                    ]
        ydd_template = pd.DataFrame(columns=col_name)
        for idx, row in self.df.iterrows():
            trk_num = str(row["Tracking Number"]).replace(" ", "").replace("[", "").replace("]", "")
            new_row = {}
            if trk_num in self.trk_to_cust:
                cust_id, lead_shipment = self.trk_to_cust[trk_num]
                self.df.at[idx, "cust_id"] = cust_id
                self.df.at[idx, "Lead Shipment Number"] = lead_shipment
            else:
                self.df.at[idx, "cust_id"] = self._exception_handler(row)
                if self.df.at[idx, "Lead Shipment Number"] == "":
                    self.df.at[idx, "Lead Shipment Number"] = self.df.at[idx, "Tracking Number"]
                new_row = {
                    "客户代码": self.df.at[idx, "cust_id"], 
                    "转单号": self.df.at[idx, "Lead Shipment Number"], 
                    "承运商子单号": self.df.at[idx, "Tracking Number"], 
                    "客户单号(文本格式)": str(self.df.at[idx, "Tracking Number"])+"-"+ \
                        str(self.df.at[idx, "Shipment Reference Number 1"])[:30], 
                    "收货渠道": "UPS Ground", 
                    "目的地国家": "US", 
                    "件数": 1, 
                    "总实重": math.ceil(self.df.at[idx, "Billed Weight"]/2.204), 
                    "长": math.ceil(self.df.at[idx, "Billed Length"]*2.54), 
                    "宽": math.ceil(self.df.at[idx, "Billed Width"]*2.54), 
                    "高": math.ceil(self.df.at[idx, "Billed Height"]*2.54), 
                    "收件人姓名": self.df.at[idx, "Receiver Name"], 
                    "收件公司": self.df.at[idx, "Receiver Company Name"], 
                    "收件人地址一": self.df.at[idx, "Receiver Address Line 1"], 
                    "收件人地址二": self.df.at[idx, "Receiver Address Line 2"], 
                    "城市": self.df.at[idx, "Receiver City"], 
                    "省份/洲名简码": self.df.at[idx, "Receiver State"], 
                    "邮编": self.df.at[idx, "Receiver Postal"], 
                    "收件人电话": "000-000-0000", 
                    "包裹类型": "", 
                    "报关方式": "", 
                    "付税金": "", 
                    "中文品名": "", 
                    "英文品名": "", 
                    "海关编码": "", 
                    "数量": "", 
                    "币种": "", 
                    "单价": "", 
                    "总价": "", 
                    "购买保险": "", 
                    "寄件人姓名": self.df.at[idx, "Receiver Postal"], 
                    "寄件公司": self.df.at[idx, "Receiver Postal"], 
                    "寄件人地址一": self.df.at[idx, "Receiver Postal"], 
                    "寄件人地址二": self.df.at[idx, "Receiver Postal"], 
                    "寄件人城市": self.df.at[idx, "Receiver Postal"], 
                    "寄件人州名": self.df.at[idx, "Receiver Postal"], 
                    "寄件人邮编": self.df.at[idx, "Receiver Postal"], 
                    "寄件人国家": "US", 
                    "寄件电话": "000-000-0000", 
                    "货物特性": "", 
                    "收货备注": "", 
                    "预报备注": ""
                }
    def _exception_handler(self, row):

    def _tempalte_generator(self):
    

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
    
