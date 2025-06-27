import os
import shutil
from pathlib import Path

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

# loader = UpsInvLoader("/path/to/loading_folder")
# if loader.validate_format():
#     loader.extract_basic_info()
#     output_dir = loader.archive_raw_invoices("/path/to/archive_folder")


from pathlib import Path
from typing import List
import pandas as pd

class UpsInvNormalizer:
    def __init__(self, file_list: List[Path]):
        """
        :param file_list: List of raw invoice CSV file paths
        :param header_path: Full resolved path to OriHeadr.csv
        """
        self.file_list = file_list
        self.header_path = Path(__file__).resolve().parent / "data" / "mappings" / "OriHeadr.csv"
        self.raw_dataframes = []
        self.normalized_df = pd.DataFrame()
        self.col_rename = []
        self.dtype_map = {}
        self.date_cols = []

        # Load and process header mapping when initialized
        self._load_header_mapping()

    def _load_header_mapping(self):
        """Load header mapping CSV, build rename map, dtype map, and date columns."""
        headers_df = pd.read_csv(self.header_path)
        self.filtered_df = headers_df['Column'][headers_df['Flag'] == 1].copy()
        print(filtered_df)
        print(self.dtype_map)

    def load_invoices(self):
        """Load CSV invoice files with correct dtypes and dates."""
        self.raw_dataframes = []
        for file in self.file_list:
            df = pd.read_csv(
                file, 
                header=None,
                names=col_name,
                dtype=self.dtype_map
                )
            self.raw_dataframes.append(df)
    
        print(df[:5])

    def merge_invoices(self):
        """Merge loaded DataFrames into one."""
        self.normalized_df = pd.concat(self.raw_dataframes, ignore_index=True)

    def standardize_invoices(self):
        """Standardize and enrich columns as per business rules."""
        df = self.normalized_df
        col_indices = list(self.rename_map.keys())
        col_names = [self.rename_map[i] for i in col_indices]
        df = df.iloc[:, col_indices]
        df.columns = col_names

        # Convert to datetime after columns are renamed
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

        df['Account Number'] = df['Account Number'].astype(str).str[-6:]
        df['Invoice Number'] = df['Invoice Number'].astype(str).str[-9:]

        dims = df['Package Dimensions'].str.split('X', expand=True).astype(float)
        dims.columns = ['Billed Length', 'Billed Width', 'Billed Height']
        idx = df.columns.get_loc('Package Dimensions') + 1
        for i, col in enumerate(dims.columns):
            df.insert(idx + i, col, dims[col])

        dims2 = df['Place Holder 35'].str.split('X', expand=True).astype(float)
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
    


