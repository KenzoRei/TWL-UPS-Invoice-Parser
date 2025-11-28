"""UPS Invoice normalizer for standardizing and enriching invoice data."""

from pathlib import Path
from typing import List

import pandas as pd

from config import SPECIAL_CUSTOMERS, FLAG_API_USE
from utils.helpers import extract_dims


class UpsInvNormalizer:
    """Normalize and standardize UPS invoice data."""
    
    def __init__(self, file_list: List[Path]):
        """
        Initialize normalizer.
        
        Args:
            file_list: List of raw invoice CSV file paths
        """
        self.file_list = file_list
        self.header_path = Path(__file__).resolve().parent.parent / "data" / "mappings" / "OriHeadr.csv"
        self.headers = pd.DataFrame()
        self.raw_dataframes: List[pd.DataFrame] = []
        self.normalized_df = pd.DataFrame()
        self.all_col_name: List[str] = []
        self.filtered_col_name: List[str] = []
        self.date_cols: List[str] = []
        self.dtype_map: dict = {}

        # Load and process header mapping when initialized
        self._load_header_mapping()

    def _validate_header_mapping(self):
        """
        Validate header mapping file for:
        1. NaNs in critical columns
        2. Valid formats in 'Format' column
        
        Exits program if validation fails.
        """
        if self.headers['Column Name'].isna().any():
            print("❌ ERROR: 'Column Name' column in header mapping contains NaN values. "
                  "Please fix OriHeadr.csv.")
            exit(1)
            
        if self.headers['Format'].isna().any():
            print("❌ ERROR: 'Format' column in header mapping contains NaN values. "
                  "Please fix OriHeadr.csv.")
            exit(1)

        # Allowed formats
        allowed_formats = {'str', 'float', 'int', 'date'}
        invalid_formats = set(self.headers['Format'].unique()) - allowed_formats
        if invalid_formats:
            print(f"❌ ERROR: 'Format' column contains invalid values: {invalid_formats}. "
                  f"Allowed values are: {allowed_formats}. Please fix OriHeadr.csv.")
            exit(1)

    def _load_header_mapping(self):
        """
        Load header mapping CSV and build:
        - Rename map
        - Dtype map
        - Date columns list
        """
        self.headers = pd.read_csv(self.header_path)
        self._validate_header_mapping()
        
        self.all_col_name = self.headers['Column Name'].tolist()
        
        # Non-date columns with Flag == 1
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
        
        self.filtered_col_name = self.headers.loc[
            self.headers['Flag'] == 1, 'Column Name'
        ].tolist()
        
        self.date_cols = self.headers.loc[mask_date, 'Column Name'].tolist()

    def _save_trk_nums(self):
        """
        Save unique tracking numbers to output/trk_nums.csv.
        
        Excludes tracking numbers from special customer accounts.
        Only called when API is not being used.
        """
        out_path = Path(__file__).resolve().parent.parent / "output"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "trk_nums.csv"

        # Setup account list for special customers
        special_cust_acct = [
            acct
            for cust in SPECIAL_CUSTOMERS.values()
            for acct in cust["accounts"]
        ]

        # Filter all unique tracking numbers
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
        """
        Load CSV invoice files with correct dtypes and dates.
        
        Tries multiple encodings to handle various file formats.
        Parses date columns with mixed formats.
        """
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
                        encoding=enc,
                        encoding_errors="strict",
                    )
                    # Success - stop trying encodings
                    print(f"✓ Loaded {file.name} with encoding {enc}")
                    break
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
            else:
                # No encoding worked; force replacement as last resort
                df = pd.read_csv(
                    file,
                    header=None,
                    names=self.all_col_name,
                    dtype=self.dtype_map,
                    encoding="latin1",
                    encoding_errors="replace",
                )
                print(f"! Loaded {file.name} with latin1 (replacement used)")

            # Keep only flagged columns
            df = df.loc[:, self.filtered_col_name]

            # Parse date columns safely (mixed formats)
            for date_col in self.date_cols:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

            self.raw_dataframes.append(df)

    def merge_invoices(self):
        """Merge loaded DataFrames into one consolidated DataFrame."""
        self.normalized_df = pd.concat(self.raw_dataframes, ignore_index=True)

    def standardize_invoices(self):
        """
        Standardize and enrich columns according to business rules.
        
        Operations:
        - Ensure string columns are properly typed
        - Parse date columns
        - Format account and invoice numbers
        - Extract package dimensions from text fields
        - Add calculated columns (Basis Amount, charge categories)
        - Save tracking numbers if API is not used
        """
        # Ensure string columns are converted correctly
        str_cols = self.headers.loc[
            (self.headers['Flag'] == 1) & (self.headers['Format'] == 'str'), 
            'Column Name'
        ].tolist()
        for col in str_cols:
            self.normalized_df[col] = self.normalized_df[col].astype(str)

        df = self.normalized_df

        # Convert to datetime after columns are renamed
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')

        # Format account and invoice numbers
        df['Account Number'] = df['Account Number'].astype(str).str[-6:]
        df['Invoice Number'] = df['Invoice Number'].astype(str).str[-9:]

        # Parse billed (package) dimensions
        billed_dims = extract_dims(df["Package Dimensions"], "Billed")
        
        # Insert these three columns right after "Package Dimensions"
        if "Package Dimensions" in df.columns:
            insert_at = df.columns.get_loc("Package Dimensions") + 1
        else:
            insert_at = len(df.columns)
            
        for i, col in enumerate(["Billed Length", "Billed Width", "Billed Height"]):
            df.insert(insert_at + i, col, billed_dims.get(col))

        # Parse entered dimensions from "Place Holder 35"
        entered_dims = extract_dims(df["Place Holder 35"], "Entered")
        
        if "Place Holder 35" in df.columns:
            insert_at2 = df.columns.get_loc("Place Holder 35") + 1
        else:
            insert_at2 = len(df.columns)
            
        for i, col in enumerate(["Entered Length", "Entered Width", "Entered Height"]):
            df.insert(insert_at2 + i, col, entered_dims.get(col))

        # Add Basis Amount column
        insert_idx = df.columns.get_loc('Incentive Amount')
        df.insert(insert_idx, 'Basis Amount', df['Incentive Amount'] + df['Net Amount'])

        # Add charge category columns
        charge_idx = df.columns.get_loc('Charge Description') + 1
        df.insert(charge_idx, 'Charge_Cate_EN', '')
        df.insert(charge_idx + 1, 'Charge_Cate_CN', '')

        self.normalized_df = df
        
        # Save tracking numbers if not using API
        if not FLAG_API_USE:
            self._save_trk_nums()

    def get_normalized_data(self) -> pd.DataFrame:
        """
        Get the normalized DataFrame.
        
        Returns:
            Normalized and standardized DataFrame
        """
        return self.normalized_df
