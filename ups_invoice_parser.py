from __future__ import annotations

# Standard library
import csv
import datetime
import functools
import logging
import os
import pickle
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
import requests

# Local application
from models import Invoice, Shipment, Package, Charge, Location
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
    "F000222": {"accounts": {"H930G2", "H930G3", "H930G4", "R1H015", "XJ3936", "Y209J6", "Y215B9", "0H947Y"}},
    "F000208": {"accounts": {"HE6132"}},
}
SPECIAL_ACCT_TO_CUST = {acct: cust for cust, v in SPECIAL_CUSTOMERS.items() for acct in v["accounts"]}
SPECIAL_CUSTS = set(SPECIAL_CUSTOMERS.keys())
FLAG_API_USE = True
YDD_USER = "5055457@qq.com"
YDD_PASS = "Twc11434!"
FLAG_DEBUG = False

def retry_on_timeout(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0, jitter: float = 0.5, timeout_increase: float = 1.5):
    """
    Decorator for retrying functions that may encounter timeout errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        jitter: Random jitter factor (0-1) to avoid thundering herd
        timeout_increase: Factor to increase timeout on retries
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            original_timeout = kwargs.get('timeout', 20)  # Default timeout
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Increase timeout progressively on retries
                    if attempt > 0:
                        kwargs['timeout'] = original_timeout * (timeout_increase ** attempt)
                        if FLAG_DEBUG:
                            print(f"[Retry] Attempt {attempt + 1}/{max_retries + 1} with timeout {kwargs['timeout']:.1f}s")
                    
                    return func(*args, **kwargs)
                    
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, 
                        requests.exceptions.Timeout) as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        # Always show final failure message
                        print(f"⚠️  API timeout after {max_retries + 1} attempts. Network may be unstable.")
                        if FLAG_DEBUG:
                            print(f"[Retry] All {max_retries + 1} attempts failed for {func.__name__}")
                        raise e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (backoff_factor ** attempt)
                    jitter_amount = delay * jitter * random.random()
                    total_delay = delay + jitter_amount
                    
                    # Always show retry warning to user
                    print(f"⚠️  Network timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {total_delay:.1f}s...")
                    
                    if FLAG_DEBUG:
                        print(f"[Retry] {func.__name__} detailed timeout info - retrying in {total_delay:.2f}s...")
                    
                    time.sleep(total_delay)
                    
                except Exception as e:
                    # Non-timeout exceptions are not retried
                    raise e
                    
            # This should never be reached, but just in case
            raise last_exception
            
        return wrapper
    return decorator

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


# --- safe dimension parser (works even if text is missing/malformed) ---
def _extract_dims(series: pd.Series, col_prefix: str):
    # Keep only digits, dots, and 'x'; normalize case/spaces
    s = (
        series.fillna("")
            .astype(str)
            .str.strip()
            .str.replace(" ", "", regex=False)
    )
    # Extract exactly LxWxH (numbers with optional decimals); unmatched rows -> NaN
    dims = s.str.extract(
        r'(?i)^\s*(?P<L>\d+(?:\.\d+)?)x(?P<W>\d+(?:\.\d+)?)x(?P<H>\d+(?:\.\d+)?)\s*$'
    )
    # Convert to numeric
    for c in ["L", "W", "H"]:
        dims[c] = pd.to_numeric(dims[c], errors="coerce")
    # Rename to requested columns
    dims = dims.rename(columns={
        "L": f"{col_prefix} Length",
        "W": f"{col_prefix} Width",
        "H": f"{col_prefix} Height",
    })
    return dims

# -- inside UpsInvLoader --
class UpsInvLoader:
    def __init__(self, input_dir: str = "."):
        self.input_dir = Path(input_dir)
        self.invoices: list[Path] = []
        self.basic_info: list[dict] = []
        self.batch_number: str | None = None
        self._chooser = FileChooser(self.input_dir)

    def choose_files_dialog(self, *, cli_fallback: bool = True) -> list[Path]:
        self.invoices = self._chooser.pick_csvs(interactive=True, cli_fallback=cli_fallback)
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
        If files are already from archive folder, skip archiving process.
        """
        if not self.invoices:
            raise ValueError("No files selected to archive. Run choose_files_dialog() first.")

        base_path = Path(__file__).resolve().parent / "data" / "raw_invoices"
        
        # Check if all selected files are already from archive folders
        files_from_archive = []
        files_to_archive = []
        
        for file in self.invoices:
            try:
                # Check if file path contains the archive base path
                if base_path in file.resolve().parents or file.resolve().parent == base_path:
                    files_from_archive.append(file)
                else:
                    files_to_archive.append(file)
            except Exception:
                # If path resolution fails, treat as non-archive file
                files_to_archive.append(file)
        
        # If all files are from archive, skip archiving entirely
        if files_from_archive and not files_to_archive:
            print(f"📁 All {len(files_from_archive)} selected files are from archive folders - skipping archive process")
            return
        
        # If some files are from archive, show mixed selection info
        if files_from_archive:
            print(f"📁 {len(files_from_archive)} files already in archive, {len(files_to_archive)} files to be archived")
        
        # Proceed with archiving for non-archive files
        out_path = base_path / str(self.batch_number) if self.batch_number else base_path
        out_path.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for file in files_to_archive:
            destination = out_path / file.name
            # Skip copying if the file is already in the destination (same path)
            if file.resolve() != destination.resolve():
                shutil.copy(file, destination)
                copied_count += 1

        if copied_count > 0:
            print(f"📁 Archived {copied_count} files to {out_path}")
        elif files_to_archive:
            print(f"📁 All {len(files_to_archive)} files were already in {out_path}")
        else:
            print(f"📁 No new files to archive")
    
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
                self.choose_files_dialog(cli_fallback=cli_fallback)
            except Exception:
                if not cli_fallback:
                    raise
                self.choose_files_cli()
        if not self.invoices:
            if interactive and not cli_fallback:
                # User likely clicked Cancel in dialog - provide friendly message
                print("❌ File selection was cancelled. No files were selected.")
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
    
    def _convert_dtypes_safely(self, df):
        """Convert data types safely, handling conversion errors gracefully."""
        for col_name, target_type in self.dtype_map.items():
            if col_name in df.columns:
                if target_type == int:
                    # For integer columns, first convert to float, then to int (handles decimal strings)
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    # Only convert to int if there are no decimals, otherwise keep as float
                    if df[col_name].notna().any():
                        if (df[col_name] % 1 == 0).all():
                            df[col_name] = df[col_name].astype('Int64')  # Nullable integer
                        # If there are decimals, keep as float (don't force to int)
                elif target_type == float:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                elif target_type == str:
                    df[col_name] = df[col_name].astype(str)
    
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
                if FLAG_DEBUG:
                    print(f"[DEBUG] Trying to load {file.name} with encoding {enc}...")
                try:
                    # First try with original dtype mapping
                    df = pd.read_csv(
                        file,
                        header=None,
                        names=self.all_col_name,
                        dtype=self.dtype_map,
                        low_memory=False,
                        encoding=enc,                 # ← try encodings
                        encoding_errors="strict",     # or "replace" to keep going
                    )
                    # success -> stop trying encodings
                    if FLAG_DEBUG:
                        print(f"[DEBUG] Successfully loaded {file.name} with encoding {enc}")
                    break
                except (UnicodeDecodeError, ValueError) as e:
                    if isinstance(e, UnicodeDecodeError):
                        last_err = e
                        continue
                    elif isinstance(e, ValueError) and "cannot safely convert" in str(e):
                        # Try loading without dtype constraints for problematic columns
                        if FLAG_DEBUG:
                            print(f"[DEBUG] Dtype conversion error, trying flexible loading...")
                        try:
                            df = pd.read_csv(
                                file,
                                header=None,
                                names=self.all_col_name,
                                dtype=str,  # Load everything as strings first
                                low_memory=False,
                                encoding=enc,
                                encoding_errors="strict",
                            )
                            # Convert types manually after loading
                            self._convert_dtypes_safely(df)
                            if FLAG_DEBUG:
                                print(f"[DEBUG] Loaded {file.name} with encoding {enc} (flexible mode)")
                            break
                        except UnicodeDecodeError:
                            last_err = e
                            continue
                    else:
                        raise e
            else:
                # no encoding worked; as a last resort, force replacement
                df = pd.read_csv(
                    file,
                    header=None,
                    names=self.all_col_name,
                    dtype=str,  # Use strings to avoid conversion errors
                    encoding="latin1",
                    encoding_errors="replace",  # keeps data, replaces bad bytes
                )
                self._convert_dtypes_safely(df)
                if FLAG_DEBUG:
                    print(f"[DEBUG] Loaded {file.name} with latin1 (replacement used)")

            # keep only flagged columns
            df = df.loc[:, self.filtered_col_name]

            # parse date columns safely (mixed formats)
            for date_col in self.date_cols:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

            self.raw_dataframes.append(df)
        
        # Overall loading summary
        print(f"✅ Successfully loaded {len(self.raw_dataframes)} invoice files")

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

        # Parse billed (package) dimensions
        billed_dims = _extract_dims(df["Package Dimensions"], "Billed")
        # Insert these three columns right after "Package Dimensions"
        if "Package Dimensions" in df.columns:
            insert_at = df.columns.get_loc("Package Dimensions") + 1
        else:
            insert_at = len(df.columns)
        for i, col in enumerate(["Billed Length", "Billed Width", "Billed Height"]):
            df.insert(insert_at + i, col, billed_dims.get(col))

        # Parse entered dimensions from "Place Holder 35"
        entered_dims = _extract_dims(df["Place Holder 35"], "Entered")
        if "Place Holder 35" in df.columns:
            insert_at2 = df.columns.get_loc("Place Holder 35") + 1
        else:
            insert_at2 = len(df.columns)
        for i, col in enumerate(["Entered Length", "Entered Width", "Entered Height"]):
            df.insert(insert_at2 + i, col, entered_dims.get(col))

        insert_idx = df.columns.get_loc('Incentive Amount')
        df.insert(insert_idx, 'Basis Amount', df['Incentive Amount'] + df['Net Amount'])

        charge_idx = df.columns.get_loc('Charge Description') + 1
        df.insert(charge_idx, 'Charge_Cate_EN', '')
        df.insert(charge_idx + 1, 'Charge_Cate_CN', '')

        self.normalized_df = df
        if not FLAG_API_USE:
            self._save_trk_nums()

    def get_normalized_data(self) -> pd.DataFrame:
        return self.normalized_df

# -----------------------------
# UpsCustomerMatcher (updated for YDD API)
# -----------------------------
class UpsCustomerMatcher:
    def __init__(
        self,
        normalized_df: pd.DataFrame,
        mapping_file: Path | None = None,
        *,
        use_api: bool = FLAG_API_USE,
        ydd_threads: int = 1,         # 1 = sequential; >1 enables parallel
        ydd_batch_size: int = 9,      # API limit is 10
        ydd_client: Optional[object] = None,
        use_cache: bool = True,       # optional on-disk cache for API mapping
    ):
        """
        :param normalized_df: DataFrame output from normalizer.get_normalized_data()
        :param mapping_file:  数据列表*.xlsx (only used when use_api=False)
        :param use_api:       True -> fetch mapping via YDD API; False -> use manual Excel mapping
        :param ydd_threads:   Parallel threads for YDD API (1 = sequential)
        :param ydd_batch_size:Max refs per request (YDD docs say 10)
        :param ydd_client:    Optional preconfigured YDDClient; if None, env vars are used
        :param use_cache:     If True, cache danHao→(cust_id,tracking) to output/ydd_ref_map.csv
        """
        self.df = normalized_df.copy()
        if "cust_id" not in self.df.columns: self.df["cust_id"] = ""
        if "Charge_Cate_EN" not in self.df.columns: self.df["Charge_Cate_EN"] = ""
        if "Charge_Cate_CN" not in self.df.columns: self.df["Charge_Cate_CN"] = ""

        self.DEFAULT_CUST_ID = "F000999"
        self.base_path = Path(__file__).resolve().parent

        # mode / knobs
        self.use_api = use_api
        self.ydd_threads = max(1, int(ydd_threads))
        self.ydd_batch_size = max(1, int(ydd_batch_size))
        self._ydd_client = ydd_client
        self.use_cache = use_cache

        # manual mapping fields
        self.mapping_file: Path | None = mapping_file
        self.mapping_cust_df = pd.DataFrame()
        self.trk_to_cust: Dict[str, Tuple[str, str]] = {}   # Tracking -> (cust_id, LeadShipment)

        # API mapping fields
        self.ref_to_cust: Dict[str, Tuple[str, str]] = {}   # danHao -> (cust_id, chosen_tracking)

        # shared mappings (csv)
        self.mapping_pickup = self.base_path / "data" / "mappings" / "Pickups.csv"
        self.mapping_chrg   = self.base_path / "data" / "mappings" / "Charges.csv"
        self.mapping_ar     = self.base_path / "data" / "mappings" / "ARCalculator.csv"

        self.mapping_pickup_df = pd.DataFrame()
        self.mapping_chrg_df   = pd.DataFrame()
        self.mapping_ar_df     = pd.DataFrame()

        self.dict_pickup: Dict[str, dict] = {}
        self.dict_chrg:   Dict[str, dict] = {}
        self.dict_ar:     Dict[str, dict] = {}

        # New cache configuration (data/cache/ structure)
        self.cache_dir = self.base_path / "data" / "cache"
        self.cache_file = self.cache_dir / "trk_to_cust.csv"
        self.archive_dir = self.cache_dir / "archive"
        self.max_cache_records = 100000  # 100K record limit
        
        # Legacy cache file (will be migrated)
        self.api_cache_path = self.base_path / "output" / "ydd_ref_map.csv"

        # exception export
        self.excluded_from_exception = set(SPECIAL_ACCT_TO_CUST.keys())

        self.api_stats: dict = {}

    # ---------------- small helpers ----------------
    def _best_tracking_for_row(self, row: pd.Series) -> str:
        ls  = str(row.get("Lead Shipment Number", "") or "").strip()
        trk = str(row.get("Tracking Number", "") or "").strip()
        return ls if not is_blank(ls) else trk

    # ---------------- cache management ----------------
    def _ensure_cache_structure(self) -> None:
        """Create cache directory structure if needed"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_trk2cust_cache(self) -> None:
        """Create empty cache file with proper headers"""
        self._ensure_cache_structure()
        
        # Create empty DataFrame with proper columns
        empty_cache = pd.DataFrame(columns=[
            'tracking', 'cust_id', 'transfer_no', 'created_at', 'source'
        ])
        empty_cache.to_csv(self.cache_file, index=False, encoding='utf-8-sig')
        print(f"[Cache] Created empty cache file: {self.cache_file}")
        
    def _load_trk2cust_cache(self) -> Dict[str, Tuple[str, str]]:
        """Load trk→(cust_id, transfer_no) from data/cache/trk_to_cust.csv"""
        if not self.use_cache:
            return {}
            
        # Create cache structure if needed
        self._ensure_cache_structure()
        
        # Create cache file if it doesn't exist
        if not self.cache_file.exists():
            self._create_trk2cust_cache()
            return {}
            
        try:
            cache_df = pd.read_csv(self.cache_file, encoding='utf-8-sig')
        except pd.errors.EmptyDataError:
            print(f"[Cache] Cache file is empty, recreating: {self.cache_file}")
            self._create_trk2cust_cache()
            return {}
        except Exception as e:
            print(f"[Cache] Failed to read cache {self.cache_file}: {e}")
            return {}
            
        # Validate required columns
        required_cols = {'tracking', 'cust_id', 'transfer_no'}
        if not required_cols.issubset(cache_df.columns):
            missing = required_cols - set(cache_df.columns)
            print(f"[Cache] Cache missing columns {missing}, recreating")
            self._create_trk2cust_cache()
            return {}
            
        # Convert to dictionary format: tracking → (cust_id, transfer_no)
        cache_dict = {}
        for _, row in cache_df.iterrows():
            tracking = str(row['tracking']).strip()
            cust_id = str(row['cust_id']).strip()
            transfer_no = str(row['transfer_no']).strip()
            
            if tracking and cust_id:  # Only add valid entries
                cache_dict[tracking] = (cust_id, transfer_no)
        if FLAG_DEBUG:
            print(f"[Debug] Loaded {len(cache_dict)} mappings from cache")
        return cache_dict
        
    def _update_trk2cust_cache(self, new_mappings: Dict[str, Tuple[str, str]]) -> None:
        """Update cache with new mappings (basic version without archiving)"""
        if not self.use_cache or not new_mappings:
            return
            
        # Load existing cache
        cache_dict = self._load_trk2cust_cache()
        
        # Add new mappings
        now = datetime.datetime.now().isoformat()
        updated_count = 0
        
        for tracking, (cust_id, transfer_no) in new_mappings.items():
            if tracking.strip() and cust_id.strip():
                cache_dict[tracking] = (cust_id, transfer_no)
                updated_count += 1
                
        # Convert back to DataFrame
        cache_rows = []
        for tracking, (cust_id, transfer_no) in cache_dict.items():
            cache_rows.append({
                'tracking': tracking,
                'cust_id': cust_id,
                'transfer_no': transfer_no,
                'created_at': now,
                'source': 'api'
            })
            
        cache_df = pd.DataFrame(cache_rows)
        
        # Check cache size and auto-archive if needed
        if len(cache_df) > self.max_cache_records:
            cache_df = self._archive_cache_overflow(cache_df)
            
        # Save updated cache
        try:
            # Use temp file for atomic writes
            temp_file = self.cache_file.with_suffix('.tmp')
            cache_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            temp_file.replace(self.cache_file)
            if FLAG_DEBUG:
                print(f"[Debug] Updated cache: {len(cache_df)} total records (+{updated_count} new)")
        except Exception as e:
            print(f"[Cache] Failed to save cache: {e}")
            
    def _archive_cache_overflow(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        """
        Archive overflow cache records when limit is exceeded
        
        Args:
            cache_df: Current cache DataFrame
            
        Returns:
            pd.DataFrame: Trimmed cache DataFrame (most recent records)
        """
        if len(cache_df) <= self.max_cache_records:
            return cache_df
            
        print(f"[Cache] Cache size {len(cache_df)} exceeds limit {self.max_cache_records}")
        print(f"[Cache] Starting automatic archiving process...")
        
        try:
            # Create archive directory if needed
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Sort by created_at to keep most recent records
            if 'created_at' in cache_df.columns:
                cache_df_sorted = cache_df.sort_values('created_at', ascending=False)
            else:
                # If no timestamp, keep last records (DataFrame order)
                cache_df_sorted = cache_df.copy()
                
            # Split into keep vs archive
            keep_records = self.max_cache_records // 2  # Keep 50K most recent
            cache_to_keep = cache_df_sorted.head(keep_records).copy()
            cache_to_archive = cache_df_sorted.tail(len(cache_df_sorted) - keep_records).copy()
            
            # Generate archive filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"trk_to_cust_archive_{timestamp}.csv"
            archive_path = self.archive_dir / archive_filename
            
            # Save archived records
            cache_to_archive.to_csv(archive_path, index=False, encoding='utf-8-sig')
            
            print(f"[Cache] ✅ Archived {len(cache_to_archive)} old records to: {archive_filename}")
            print(f"[Cache] ✅ Keeping {len(cache_to_keep)} most recent records in active cache")
            
            # Update archive statistics
            archive_stats = {
                'archive_date': timestamp,
                'archived_count': len(cache_to_archive),
                'kept_count': len(cache_to_keep),
                'archive_file': archive_filename
            }
            
            # Optional: Save archive metadata
            metadata_path = self.archive_dir / "archive_log.csv"
            metadata_df = pd.DataFrame([archive_stats])
            
            if metadata_path.exists():
                # Append to existing log
                existing_metadata = pd.read_csv(metadata_path)
                metadata_df = pd.concat([existing_metadata, metadata_df], ignore_index=True)
                
            metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
            
            return cache_to_keep
            
        except Exception as e:
            print(f"[Cache] ❌ Archive process failed: {e}")
            print(f"[Cache] ❌ Continuing with full cache (may impact performance)")
            return cache_df

    def _migrate_legacy_cache(self) -> Dict[str, Tuple[str, str]]:
        """Migrate data from old ydd_ref_map.csv to new cache format (one-time migration)"""
        legacy_mappings = {}
        
        # Try to load from legacy cache
        if self.api_cache_path.exists():
            try:
                legacy_df = pd.read_csv(self.api_cache_path)
                required = {"danHao", "cust_id", "tracking"}
                
                if required.issubset(legacy_df.columns):
                    # Convert danHao mappings to tracking mappings
                    for _, row in legacy_df.iterrows():
                        tracking = str(row["tracking"]).strip()
                        cust_id = str(row["cust_id"]).strip()
                        danhao = str(row["danHao"]).strip()
                        
                        if tracking and cust_id:
                            legacy_mappings[tracking] = (cust_id, danhao)
                            
                    print(f"[Cache] Migrated {len(legacy_mappings)} mappings from legacy cache")
                    
                    # Save to new cache format
                    if legacy_mappings:
                        self._update_trk2cust_cache(legacy_mappings)
                        
            except Exception as e:
                print(f"[Cache] Failed to migrate legacy cache: {e}")
                
        return legacy_mappings

    def manage_cache(self, action: str = "status") -> dict:
        """
        Manual cache management interface
        
        Args:
            action: "status", "archive", "clear", "stats"
            
        Returns:
            dict: Cache management results and statistics
        """
        if not self.use_cache:
            return {"error": "Cache is disabled (use_cache=False)"}
            
        try:
            # Ensure cache structure exists
            self._ensure_cache_structure()
            
            if action == "status":
                # Get current cache status
                if not self.cache_file.exists():
                    return {
                        "status": "no_cache",
                        "message": "Cache file does not exist",
                        "cache_file": str(self.cache_file)
                    }
                    
                try:
                    cache_df = pd.read_csv(self.cache_file, encoding='utf-8-sig')
                    total_records = len(cache_df)
                    size_mb = self.cache_file.stat().st_size / (1024 * 1024)
                    
                    # Check for archive files
                    archive_files = list(self.archive_dir.glob("trk_to_cust_archive_*.csv"))
                    
                    return {
                        "status": "active",
                        "total_records": total_records,
                        "max_records": self.max_cache_records,
                        "size_mb": round(size_mb, 2),
                        "usage_percent": round((total_records / self.max_cache_records) * 100, 1),
                        "archive_files": len(archive_files),
                        "needs_archiving": total_records > self.max_cache_records,
                        "cache_file": str(self.cache_file),
                        "archive_dir": str(self.archive_dir)
                    }
                except Exception as e:
                    return {"error": f"Failed to read cache: {e}"}
                    
            elif action == "archive":
                # Force archive current cache
                if not self.cache_file.exists():
                    return {"error": "Cache file does not exist"}
                    
                cache_df = pd.read_csv(self.cache_file, encoding='utf-8-sig')
                original_size = len(cache_df)
                
                if original_size <= self.max_cache_records // 2:
                    return {"message": f"Cache size ({original_size}) is already below archive threshold"}
                    
                # Force archive
                archived_df = self._archive_cache_overflow(cache_df)
                
                return {
                    "status": "archived",
                    "original_records": original_size,
                    "remaining_records": len(archived_df),
                    "archived_records": original_size - len(archived_df)
                }
                
            elif action == "clear":
                # Clear entire cache (use with caution)
                if self.cache_file.exists():
                    backup_name = f"trk_to_cust_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    backup_path = self.archive_dir / backup_name
                    
                    # Backup before clearing
                    import shutil
                    shutil.copy2(self.cache_file, backup_path)
                    
                    # Create fresh cache
                    self._create_trk2cust_cache()
                    
                    return {
                        "status": "cleared",
                        "message": "Cache cleared successfully",
                        "backup_created": str(backup_path)
                    }
                else:
                    return {"message": "Cache file did not exist"}
                    
            elif action == "stats":
                # Detailed cache statistics
                stats = self.manage_cache("status")
                
                if "error" in stats:
                    return stats
                    
                # Add timing and performance stats
                if self.cache_file.exists():
                    mtime = datetime.datetime.fromtimestamp(self.cache_file.stat().st_mtime)
                    stats["last_modified"] = mtime.isoformat()
                    stats["last_modified_ago"] = str(datetime.datetime.now() - mtime)
                
                # Archive directory stats
                if self.archive_dir.exists():
                    archive_files = list(self.archive_dir.glob("*.csv"))
                    total_archive_size = sum(f.stat().st_size for f in archive_files)
                    stats["archive_files_detail"] = len(archive_files)
                    stats["archive_size_mb"] = round(total_archive_size / (1024 * 1024), 2)
                    
                return stats
                
            else:
                return {"error": f"Unknown action: {action}. Use 'status', 'archive', 'clear', or 'stats'"}
                
        except Exception as e:
            return {"error": f"Cache management failed: {e}"}

    def set_mapping_file(self, path: Path) -> None:
        self.mapping_file = Path(path) if path else None

    def choose_mapping_file_dialog(self) -> Path | None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise RuntimeError("tkinter is not available for file selection.") from e
        root = tk.Tk()
        root.withdraw()
        
        # Force dialog to appear on top
        root.attributes('-topmost', True)
        root.focus_force()
        root.lift()
        
        fp = filedialog.askopenfilename(
            title="选择数据列表*.xlsx 映射文件 / Select mapping file",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        root.update(); root.destroy()
        self.mapping_file = Path(fp) if fp else None
        return self.mapping_file

    # ---------------- common CSV mappings ----------------
    def _load_common_mappings(self) -> None:
        self.mapping_chrg_df = pd.read_csv(self.mapping_chrg)
        self.dict_chrg = self.mapping_chrg_df.set_index(self.mapping_chrg_df.columns[0]).to_dict(orient="index")

        self.mapping_pickup_df = pd.read_csv(self.mapping_pickup)
        self.dict_pickup = self.mapping_pickup_df.set_index(self.mapping_pickup_df.columns[0]).to_dict(orient="index")

        self.mapping_ar_df = pd.read_csv(self.mapping_ar)
        self.dict_ar = self.mapping_ar_df.set_index(self.mapping_ar_df.columns[0]).to_dict(orient="index")

    # ---------------- manual Excel path ----------------
    def _load_mapping_manual(self) -> None:
        if self.mapping_file is None:
            raise FileNotFoundError("Mapping file not set. Call choose_mapping_file_dialog() or set_mapping_file().")
        self.mapping_cust_df = pd.read_excel(self.mapping_file)
        required_cols = {"子转单号", "客户编号", "转单号"}
        if not required_cols.issubset(self.mapping_cust_df.columns):
            missing = required_cols - set(self.mapping_cust_df.columns)
            raise ValueError(f"Mapping file missing required columns: {missing}")

        df = self.mapping_cust_df.copy()
        df["子转单号"] = (
            df["子转单号"].astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        df = df.assign(Tracking=df["子转单号"].str.split(",")).explode("Tracking")
        df["Tracking"] = df["Tracking"].astype(str).str.replace(" ", "", regex=False)
        df = df[df["Tracking"] != ""]

        # Tracking -> (cust, LeadShipment/转单号)
        self.trk_to_cust = dict(
            zip(df["Tracking"], zip(df["客户编号"].astype(str), df["转单号"].astype(str)))
        )
        self._load_common_mappings()

    # ---------------- API path ----------------
    def _ensure_ydd_client(self):
        if self._ydd_client is not None:
            return self._ydd_client
        base = os.environ.get("YDD_BASE", "http://twc.itdida.com/itdida-api")
        user = os.environ.get("YDD_USER", YDD_USER)
        pwd  = os.environ.get("YDD_PASS", YDD_PASS)
        if not user or not pwd:
            raise RuntimeError("YDD creds missing. Set env YDD_USER and YDD_PASS (opt YDD_BASE).")
        from YDD_Client import YDDClient
        self._ydd_client = YDDClient(base=base, username=user, password=pwd)
        return self._ydd_client

    def _call_ydd_api_with_retry(self, api_method, *args, **kwargs):
        """
        Wrapper for YDD API calls with enhanced retry logic for timeout handling.
        """
        @retry_on_timeout(max_retries=3, base_delay=2.0, backoff_factor=2.0, jitter=0.3)
        def _api_call_with_timeout(timeout=30):
            # Set timeout for the underlying requests in YDD client
            original_timeout = getattr(api_method.__self__, '_timeout', None)
            try:
                # Try to set timeout on the client if supported
                if hasattr(api_method.__self__, '_timeout'):
                    api_method.__self__._timeout = timeout
                elif hasattr(api_method.__self__, 'session'):
                    # If there's a session, set timeout there
                    api_method.__self__.session.timeout = timeout
                
                return api_method(*args, **kwargs)
            finally:
                # Restore original timeout
                if original_timeout is not None and hasattr(api_method.__self__, '_timeout'):
                    api_method.__self__._timeout = original_timeout

        try:
            return _api_call_with_timeout()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, 
                requests.exceptions.Timeout) as e:
            print(f"❌ YDD API connection failed after multiple retries")
            print(f"💡 Possible solutions:")
            print(f"   • Check internet connection")
            print(f"   • Verify twc.itdida.com is accessible")
            print(f"   • Try again in a few minutes (server may be busy)")
            if FLAG_DEBUG:
                print(f"[Debug] Full error: {e}")
            raise e

    def _collect_danhaos_with_tracking(self) -> tuple[list[str], dict[str, str]]:
        """
        Build:
          - danhaos: unique Shipment Reference Number 1 values
          - ref_to_best_trk: danHao -> chosen_tracking (LeadShipment preferred, else Tracking)
        Only include rows where Account Number is NOT a special account.
        """
        df = self.df.copy()
        specials = set(SPECIAL_ACCT_TO_CUST.keys())

        if FLAG_DEBUG:
            print(f"[DEBUG] Specials (excluded from YDD): {specials}")
        mask = ~df["Account Number"].astype(str).isin(specials)
        excluded_accounts = df.loc[~mask, "Account Number"].unique().tolist()
        if FLAG_DEBUG:
            print(f"[DEBUG] Excluded Account Numbers: {excluded_accounts}")
        included_accounts = df.loc[mask, "Account Number"].unique().tolist()
        if FLAG_DEBUG:
            print(f"[DEBUG] Included Account Numbers (sent to YDD): {included_accounts}")

        sub = df.loc[mask, ["Shipment Reference Number 1", "Lead Shipment Number", "Tracking Number"]].copy()
        sub["Shipment Reference Number 1"] = sub["Shipment Reference Number 1"].astype(str).str.strip()
        sub = sub[sub["Shipment Reference Number 1"].apply(lambda x: not is_blank(x))]

        sub["best_trk"] = sub.apply(self._best_tracking_for_row, axis=1)
        sub = sub.drop_duplicates(subset=["Shipment Reference Number 1"], keep="first")
        
        refs = sub["Shipment Reference Number 1"].tolist()
        if FLAG_DEBUG:            
            df.loc[mask].to_excel(
                self.base_path / "output" / "ydd_refs_sent.xlsx"
                , index=False)
            print(f"[DEBUG] Shipment sent to YDD saved to output/ydd_refs_sent.xlsx")
        ref_to_best_trk = dict(zip(sub["Shipment Reference Number 1"], sub["best_trk"]))
        return refs, ref_to_best_trk

    # ---------------- enhanced matching workflow ----------------
    def _collect_trk_ref(self) -> Dict[str, str]:
        """
        Create tracking → ref dict for enhanced matching workflow.
        Replaces _collect_danhaos_with_tracking with focus on unique tracking numbers.
        
        Returns:
            Dict[str, str]: tracking → ref mapping (tracking must be unique/non-empty, ref can be empty)
        """
        df = self.df.copy()
        specials = set(SPECIAL_ACCT_TO_CUST.keys())

        if FLAG_DEBUG:
            print(f"[DEBUG] Collecting trk->ref mappings, excluding specials: {specials}")

        # Filter out special accounts
        mask = ~df["Account Number"].astype(str).isin(specials)
        sub = df.loc[mask, ["Tracking Number", "Lead Shipment Number", "Shipment Reference Number 1"]].copy()

        # Clean and prepare tracking numbers
        sub["tracking"] = sub["Tracking Number"].astype(str).str.strip()
        sub.loc[sub["tracking"].apply(is_blank), "tracking"] = sub["Lead Shipment Number"].astype(str).str.strip()

        # Clean reference numbers (allow empty refs)
        sub["ref"] = sub["Shipment Reference Number 1"].astype(str).str.strip()
        sub.loc[sub["ref"].apply(is_blank), "ref"] = ""

        # Filter: tracking must be unique and non-empty
        sub = sub[~sub["tracking"].apply(is_blank)]
        sub = sub.drop_duplicates(subset=["tracking"], keep="first")

        trk_ref_dict = dict(zip(sub["tracking"], sub["ref"]))

        if FLAG_DEBUG:
            total_tracking = len(trk_ref_dict)
            with_refs = sum(1 for ref in trk_ref_dict.values() if ref)
            without_refs = total_tracking - with_refs
            print(f"[DEBUG] Collected {total_tracking} tracking numbers: {with_refs} with refs, {without_refs} without refs")

        return trk_ref_dict

    def _ref2cust_matching(self, refs: List[str]) -> Dict[str, Tuple[str, str]]:
        """
        Get ref → (cust_id, transfer_no) mapping via YDD query_yundan_detail API
        
        Args:
            refs: List of reference numbers (danHaos) to query
            
        Returns:
            Dict[str, Tuple[str, str]]: ref → (cust_id, transfer_no) mapping
        """
        if not refs:
            return {}

        # Remove empty/blank refs
        clean_refs = [str(ref).strip() for ref in refs if not is_blank(str(ref))]
        
        if not clean_refs:
            return {}
        
        if FLAG_DEBUG:
            print(f"[Debug] Querying {len(clean_refs)} refs via query_yundan_detail...")

        try:
            client = self._ensure_ydd_client()
            
            # Use existing concurrent query or sequential based on settings
            if self.ydd_threads > 1:
                api_items = self._query_concurrent(
                    client, clean_refs,
                    batch_size=min(self.ydd_batch_size, 10),
                    workers=self.ydd_threads,
                )
            else:
                # Use retry wrapper for single-threaded API calls
                api_items = self._call_ydd_api_with_retry(
                    client.query_yundan_detail,
                    clean_refs, 
                    batch_size=min(self.ydd_batch_size, 10), 
                    sleep=0.01
                )

            # Convert API response to ref -> cust mapping
            from YDD_Client import build_ref_to_cust
            ref_to_cust = build_ref_to_cust(api_items)
            if FLAG_DEBUG:
                print(f"[Debug] query_yundan_detail: {len(api_items)} API items -> {len(ref_to_cust)} mappings")

            return ref_to_cust

        except Exception as e:
            error_msg = str(e)
            print(f"[YDD] Error in _ref2cust_matching: {error_msg}")
            
            # Specific handling for 403 Forbidden errors
            if "403" in error_msg and "Forbidden" in error_msg:
                print(f"[YDD] 403 Forbidden Error Analysis:")
                print(f"   • This may be caused by invalid characters in references")
                print(f"   • Try reducing batch_size or checking reference formats")
                print(f"   • Some references: {clean_refs[:3]}...")
                print(f"   • Total references in batch: {len(clean_refs)}")
                
                # Attempt smaller batch or individual queries for debugging
                if len(clean_refs) > 1:
                    print(f"[YDD] Attempting smaller batch retry...")
                    try:
                        # Try first reference alone with retry
                        client = self._ensure_ydd_client()
                        test_items = self._call_ydd_api_with_retry(
                            client.query_yundan_detail, [clean_refs[0]], batch_size=1
                        )
                        print(f"[YDD] ✅ Single reference test successful")
                        
                        # If single works, try smaller batches
                        from YDD_Client import build_ref_to_cust
                        all_items = []
                        for ref in clean_refs[:3]:  # Test first 3 only
                            try:
                                items = self._call_ydd_api_with_retry(
                                    client.query_yundan_detail, [ref], batch_size=1
                                )
                                all_items.extend(items)
                            except Exception as single_e:
                                print(f"[YDD] ❌ Problematic ref: {ref} - {single_e}")
                        
                        if all_items:
                            result = build_ref_to_cust(all_items)
                            print(f"[YDD] 🔄 Partial recovery: {len(result)} mappings from single queries")
                            return result
                            
                    except Exception as retry_e:
                        print(f"[YDD] Retry also failed: {retry_e}")
            
            return {}

    def _trk2ref_matching(self, trackings: List[str]) -> Dict[str, str]:
        """
        Get tracking → ref mapping via YDD query_piece_detail API
        Extracts keHuDanHao from piece detail results
        
        Args:
            trackings: List of tracking numbers to query
            
        Returns:
            Dict[str, str]: tracking → keHuDanHao mapping
        """
        if not trackings:
            return {}

        # Remove empty/blank trackings
        clean_trackings = [str(trk).strip() for trk in trackings if not is_blank(str(trk))]
        
        if not clean_trackings:
            return {}

        if FLAG_DEBUG:
            print(f"[Debug] Querying {len(clean_trackings)} trackings via query_piece_detail...")

        try:
            client = self._ensure_ydd_client()

            # Query piece details with retry wrapper
            api_items = self._call_ydd_api_with_retry(
                client.query_piece_detail,
                clean_trackings, 
                batch_size=min(self.ydd_batch_size, 10), 
                sleep=0.01
            )

            # Extract tracking → keHuDanHao mapping
            trk_to_ref = {}
            for item in api_items:
                # Use the select_tracking function to get the best tracking identifier
                from YDD_Client import select_tracking
                trk = select_tracking(item)
                ke_hu_dan_hao = str(item.get("keHuDanHao", "")).strip()
                
                if trk and ke_hu_dan_hao:
                    trk_to_ref[trk] = ke_hu_dan_hao

            if FLAG_DEBUG:
                print(f"[Debug] query_piece_detail: {len(api_items)} API items -> {len(trk_to_ref)} trk->ref mappings")

            return trk_to_ref

        except Exception as e:
            print(f"[YDD] Error in _trk2ref_matching: {e}")
            return {}

    def _trk2cust_two_step_matching(self, trackings: List[str]) -> Dict[str, Tuple[str, str]]:
        """
        Two-step matching: tracking → ref → (cust_id, transfer_no)
        Combines _trk2ref_matching and _ref2cust_matching
        
        Args:
            trackings: List of tracking numbers for two-step matching
            
        Returns:
            Dict[str, Tuple[str, str]]: tracking → (cust_id, transfer_no) mapping
        """
        if not trackings:
            return {}

        if FLAG_DEBUG:
            print(f"[Debug] Starting two-step matching for {len(trackings)} trackings...")

        # Step 1: tracking → ref via piece_detail
        trk_to_ref = self._trk2ref_matching(trackings)

        if not trk_to_ref and FLAG_DEBUG:
            print("[Debug] Two-step matching: No tracking->ref mappings found")
            return {}

        # Step 2: ref → cust via yundan_detail  
        unique_refs = list(set(trk_to_ref.values()))
        ref_to_cust = self._ref2cust_matching(unique_refs)

        if not ref_to_cust and FLAG_DEBUG:
            print("[Debug] Two-step matching: No ref→cust mappings found")
            return {}

        # Step 3: Combine tracking → ref → cust
        trk_to_cust = {}
        for trk, ref in trk_to_ref.items():
            if ref in ref_to_cust:
                trk_to_cust[trk] = ref_to_cust[ref]

        if FLAG_DEBUG:
            print(f"[Debug] Two-step matching: {len(trk_to_cust)} final tracking->cust mappings")

        return trk_to_cust

    @staticmethod
    def _query_concurrent(
        client, danhaos: List[str], *, batch_size: int, workers: int,
        max_retries: int = 6, base_sleep: float = 0.5, jitter: float = 0.3
    ) -> List[dict]:
        """
        Parallel /queryYunDanDetail with per-thread Session and retry/backoff.
        """
        batches = [danhaos[i:i+batch_size] for i in range(0, len(danhaos), batch_size)]
        if not batches: return []
        if not client.token:
            raise RuntimeError("Call client.login() before _query_concurrent().")

        auth_header = {"Authorization": f"Bearer {client.token}"}
        base_url = client.base.rstrip("/") + "/queryYunDanDetail"
        default_timeout = 30  # Increased from 20 to 30 seconds

        def fetch_chunk(chunk: List[str]) -> List[dict]:
            params = {"danHaos": ",".join(chunk)}
            s = requests.Session()
            s.headers.update(auth_header)
            attempt = 0
            while True:
                try:
                    r = s.get(base_url, params=params, timeout=default_timeout)
                    if r.status_code == 401:
                        client.login()
                        s.headers["Authorization"] = f"Bearer {client.token}"
                        attempt += 1
                        continue
                    if r.status_code in (502, 503, 504, 429):
                        if attempt >= max_retries:
                            r.raise_for_status()
                        sleep = (base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
                        time.sleep(sleep)
                        attempt += 1
                        continue
                    r.raise_for_status()
                    js = r.json()
                    if not js.get("success", False):
                        return []
                    data = js.get("data") or []
                    return data if isinstance(data, list) else []
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, 
                        requests.exceptions.Timeout) as e:
                    if attempt >= max_retries:
                        print(f"⚠️  API batch timeout after {max_retries + 1} attempts. Continuing with other batches...")
                        if FLAG_DEBUG:
                            print(f"[YDD] Detailed timeout error: {e}")
                        raise
                    sleep = (base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
                    print(f"⚠️  Batch timeout (attempt {attempt + 1}/{max_retries + 1}), retrying in {sleep:.1f}s...")
                    if FLAG_DEBUG:
                        print(f"[YDD] Detailed retry info: {sleep:.2f}s delay")
                    time.sleep(sleep)
                    attempt += 1
                except requests.RequestException as e:
                    if attempt >= max_retries:
                        raise
                    sleep = (base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
                    if FLAG_DEBUG:
                        print(f"[YDD] Request error on attempt {attempt + 1}: {e}, retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    attempt += 1

        out: List[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch_chunk, b) for b in batches]
            for f in as_completed(futs):
                out.extend(f.result())
        return out

    def _load_api_cache(self) -> Dict[str, Tuple[str, str]]:
        if not self.use_cache:
            return {}
        p = self.api_cache_path
        try:
            if not p.exists() or p.stat().st_size == 0:
                return {}
        except Exception:
            return {}

        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            logging.warning(f"[YDD] Cache file is empty: {p}; ignoring.")
            return {}
        except Exception as e:
            logging.warning(f"[YDD] Failed to read cache {p}: {e}; ignoring.")
            return {}

        required = {"danHao", "cust_id", "tracking"}
        if not required.issubset(df.columns):
            logging.warning(f"[YDD] Cache missing columns {required - set(df.columns)}; ignoring.")
            return {}

        return {str(r["danHao"]): (str(r["cust_id"]), str(r["tracking"])) for _, r in df.iterrows()}

    def _save_api_cache(self, ref_to_cust: Dict[str, Tuple[str, str]]) -> None:
        if not self.use_cache:
            return
        self.api_cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"danHao": k, "cust_id": v[0], "tracking": v[1]} for k, v in ref_to_cust.items()]
        df = pd.DataFrame(rows)
        tmp = self.api_cache_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(self.api_cache_path)
    
    def _load_mapping_api(self) -> None:
        """
        Enhanced 5-step matching workflow:
        1. Load cache & collect tracking numbers  
        2. Match via references (yundan_detail API)
        3. Arrange remaining for tracking-based matching
        4. Match via two-step process (piece_detail → yundan_detail)
        5. Merge results and update cache
        """
        client = self._ensure_ydd_client()

        # Inform user about potential network delays
        print("Connecting to YDD API... (Network timeouts will be automatically retried)")

        # Login timing
        t0 = time.perf_counter()
        token = client.login()
        t1 = time.perf_counter()
        if FLAG_DEBUG:
            print(f"[Debug] Login OK in {t1 - t0:0.3f}s (token len={len(token)})")

        # ===== Step 1: Load cache & collect tracking numbers =====
        print("\n[YDD] Step 1: Loading cache and collecting tracking numbers...")
        cache_dict = self._load_trk2cust_cache()  # tracking → (cust_id, transfer_no)
        trk_ref_map = self._collect_trk_ref()     # tracking → ref
        
        total_trackings = len(trk_ref_map)
        cache_hits = sum(1 for trk in trk_ref_map if trk in cache_dict)
        unmatched_trackings = [trk for trk in trk_ref_map if trk not in cache_dict]
        
        if FLAG_DEBUG:
            print(f"[Debug] Total trackings: {total_trackings}")
            print(f"[Debug] Cache hits: {cache_hits}")
            print(f"[Debug] Unmatched trackings: {len(unmatched_trackings)}")

        # ===== Step 2: Reference-based matching (yundan_detail) =====
        print("\n[YDD] Step 2: Reference-based matching...")
        unmatched_refs = {trk_ref_map[trk] for trk in unmatched_trackings if trk_ref_map[trk]}
        unmatched_refs = [ref for ref in unmatched_refs if ref]  # Remove empty refs
        
        if FLAG_DEBUG:
            print(f"[Debug] Unique refs to query: {len(unmatched_refs)}")
        
        ref2cust = {}
        first_part_trk2cust = {}
        
        if unmatched_refs:
            ref2cust = self._ref2cust_matching(unmatched_refs)
            
            # Map trackings that have refs to customer info
            for trk in unmatched_trackings:
                ref = trk_ref_map.get(trk, "")
                if ref and ref in ref2cust:
                    first_part_trk2cust[trk] = ref2cust[ref]
        
        ref_based_matches = len(first_part_trk2cust)
        if FLAG_DEBUG:
            print(f"[Debug] Reference-based matches: {ref_based_matches}")

        # ===== Step 3: Arrange remaining trackings =====
        print("\n[YDD] Step 3: Arranging remaining trackings...")
        still_unmatched = [trk for trk in unmatched_trackings if trk not in first_part_trk2cust]

        if FLAG_DEBUG:
            print(f"[Debug] Trackings for two-step matching: {len(still_unmatched)}")

        # ===== Step 4: Two-step matching (piece_detail → yundan_detail) =====
        print("\n[YDD] Step 4: Two-step matching...")
        second_part_trk2cust = {}
        
        if still_unmatched:
            second_part_trk2cust = self._trk2cust_two_step_matching(still_unmatched)
        
        two_step_matches = len(second_part_trk2cust)

        if FLAG_DEBUG:
            print(f"[Debug] Two-step matches: {two_step_matches}")

        # ===== Step 5: Merge results and update cache =====
        print("\n[YDD] Step 5: Merging results and updating cache...")
        
        # Merge all new mappings
        all_new_mappings = {**first_part_trk2cust, **second_part_trk2cust}
        
        # Update cache with new mappings
        if all_new_mappings:
            self._update_trk2cust_cache(all_new_mappings)
        
        # Combine cache + new mappings for final result
        self.trk_to_cust = {**cache_dict, **all_new_mappings}
        
        # Load common mappings
        self._load_common_mappings()

        # Generate missing tracking report
        final_unmatched = [trk for trk in trk_ref_map if trk not in self.trk_to_cust]
        if final_unmatched:
            miss_path = self.base_path / "output" / "missing_trackings_ydd.csv"
            miss_path.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(final_unmatched, name="tracking").to_csv(miss_path, index=False, encoding="utf-8-sig")            
            print(f"[YDD] ❌ Missing {len(final_unmatched)} tracking(s). Saved to: {miss_path}")
        else:
            print("[YDD] All trackings matched!")
        final_mapped = cache_hits + ref_based_matches + two_step_matches

        # Enhanced statistics
        self.api_stats = {
            "total_trackings": total_trackings,
            "cache_hits": cache_hits,
            "ref_based_matches": ref_based_matches,
            "two_step_matches": two_step_matches,
            "total_new_matches": len(all_new_mappings),
            "final_mapped": final_mapped,
            "missing_count": len(final_unmatched),
            "missing_sample": final_unmatched[:10],
            "workflow_steps": {
                "cache_hit_rate": f"{cache_hits / total_trackings * 100:.1f}%" if total_trackings > 0 else "0%",
                "ref_success_rate": f"{ref_based_matches / len(unmatched_trackings) * 100:.1f}%" if unmatched_trackings else "0%",
                "two_step_success_rate": f"{two_step_matches / len(still_unmatched) * 100:.1f}%" if still_unmatched else "0%",
                "overall_success_rate": f"{final_mapped / total_trackings * 100:.1f}%" if total_trackings > 0 else "0%",
                "missing_rate": f"{len(final_unmatched) / total_trackings * 100:.1f}%" if total_trackings > 0 else "0%"
            }
        }

        print(f"\n[YDD] Enhanced workflow complete!")
        print(f"[YDD] Final stats: {cache_hits} cached + {len(all_new_mappings)} new = {final_mapped} total mappings")

    # ---------------- loader switch ----------------
    def _load_mapping(self) -> None:
        if self.use_api:
            self._load_mapping_api()
        else:
            self._load_mapping_manual()
        # For debugging: dump trk_to_cust mapping
        if FLAG_DEBUG and self.use_api:
            pd.DataFrame([
                {"tracking": k, "cust_id": v[0], "transfer_no": v[1]}
                for k, v in self.trk_to_cust.items()
            ]).to_excel("output/trk_to_cust_check.xlsx", index=False)
            print(f"[DEBUG] trk_to_cust mapping saved to output/trk_to_cust_check.xlsx")

    # ---------------- main workflow ----------------
    def match_customers(self) -> None:
        # ensure mappings are ready
        if not (self.dict_chrg and self.dict_pickup and self.dict_ar) and not self.trk_to_cust:
            self._load_mapping()

        # 0) pre-assign specials by account
        self.df["cust_id"] = self.df["cust_id"].astype(str)
        self.df["cust_id_special"] = self.df["Account Number"].map(SPECIAL_ACCT_TO_CUST)
        mask_special = self.df["cust_id_special"].notna()
        self.df.loc[mask_special, "cust_id"] = self.df.loc[mask_special, "cust_id_special"]
        self.df.drop(columns=["cust_id_special"], inplace=True)        

        exception_rows = []
        for idx, row in self.df.iterrows():
            cust_id, lead_shipment = np.nan, np.nan
            exception_flag = False

            # classify charges
            category_en, category_cn = self._charge_classifier(row)
            # normalize (strip) classification strings to avoid mismatches caused by stray spaces
            if isinstance(category_en, str):
                category_en = category_en.strip()
            if isinstance(category_cn, str):
                category_cn = category_cn.strip()
            self.df.at[idx, "Charge_Cate_EN"] = category_en
            row["Charge_Cate_EN"] = category_en
            self.df.at[idx, "Charge_Cate_CN"] = category_cn
            row["Charge_Cate_CN"] = category_cn

            # HARD RULE: general costs never go to customers
            if category_en in General_Cost_EN:
                self.df.at[idx, "cust_id"] = self.DEFAULT_CUST_ID
                row["cust_id"] = self.DEFAULT_CUST_ID
                # keep it as invoice-level cost: do NOT invent a lead shipment number
                # and do not backfill tracking
                continue

            if self.use_api:
                # Enhanced tracking-based matching workflow
                trk_num = str(row.get("Tracking Number", "") or "").strip()
                
                # Priority: if this is a pickup charge, prefer the Pickups.csv mapping for the account
                # This prevents the YDD API/cache from overriding the intended pickup-account mapping
                if category_en in ["Daily Pickup", "Daily Pickup - Fuel"] and row["Account Number"] in self.dict_pickup:
                    cust_id = self.dict_pickup.get(row["Account Number"], {}).get("Cust.ID", self.DEFAULT_CUST_ID)
                    exception_flag = True # Manual import is always needed for pickup charges
                    lead_shipment = self._best_tracking_for_row(row)
                elif not is_blank(trk_num) and trk_num in self.trk_to_cust and row["Account Number"] not in self.excluded_from_exception:
                    cust_id, chosen_trk = self.trk_to_cust[trk_num]
                    # Clean whitespace from customer ID and tracking number
                    if isinstance(cust_id, str):
                        cust_id = cust_id.strip()
                    if isinstance(chosen_trk, str):
                        chosen_trk = chosen_trk.strip()
                    lead_shipment = chosen_trk if not is_blank(chosen_trk) else self._best_tracking_for_row(row)
                else:
                    cust_id = self._exception_handler(row)
                    exception_flag = True
                    lead_shipment = self._best_tracking_for_row(row)
            else:
                # Legacy mode: direct tracking lookup without API enhancement
                trk_num = str(row.get("Tracking Number", "") or "")
                if trk_num in self.trk_to_cust:
                    cust_id, lead_shipment = self.trk_to_cust[trk_num]
                else:
                    cust_id = self._exception_handler(row)
                    exception_flag = True
                    lead_shipment = self._best_tracking_for_row(row)

            # write back
            self.df.at[idx, "cust_id"] = cust_id
            self.df.at[idx, "Lead Shipment Number"] = lead_shipment
            if is_blank(row.get("Tracking Number", "")):
                self.df.at[idx, "Tracking Number"] = lead_shipment

            # backfill when needed
            if is_blank(self.df.at[idx, "Lead Shipment Number"]) and cust_id != self.DEFAULT_CUST_ID:
                ls = str(row.get("Invoice Number", "")) + "-Daily_Pickup"
                self.df.at[idx, "Lead Shipment Number"] = ls
                self.df.at[idx, "Tracking Number"] = ls
                self.df.at[idx, "Shipment Reference Number 1"] = ls

            if exception_flag and not is_blank(self.df.at[idx, "Lead Shipment Number"]):
                exception_rows.append(self.df.loc[idx].to_dict())

        # AR calc + templates + unmapped logging (reuse your existing methods)
        self._ar_calculator()

        df_exception = pd.DataFrame(exception_rows)
        if not df_exception.empty:
            num_cols = ["总实重", "长", "宽", "高"]
            for col in df_exception.columns:
                if col in num_cols:
                    df_exception[col] = pd.to_numeric(df_exception[col], errors="coerce").fillna(0)
                else:
                    df_exception[col] = df_exception[col].replace({np.nan: ""}).replace({"nan": ""})
            self._template_generator(df_exception)

        unmapped = self.df[self.df["Charge_Cate_EN"] == "Not Defined Charge"]
        if not unmapped.empty:
            out = self.base_path / "output" / "UnmappedCharges.xlsx"
            out.parent.mkdir(parents=True, exist_ok=True)
            unmapped.to_excel(out, index=False)

        # optional: api stats summary
        if self.use_api:
            s = self.api_stats or {}
            print("\n[YDD] Enhanced API Workflow Summary")
            print("-------------------------------------------------")
            print(f"Total trackings  : {s.get('total_trackings', 0)}")
            print(f"Cache hits       : {s.get('cache_hits', 0)}")
            print(f"Ref-based matches: {s.get('ref_based_matches', 0)}")
            print(f"Two-step matches : {s.get('two_step_matches', 0)}")
            print(f"Total new matches: {s.get('total_new_matches', 0)}")
            print(f"Final mapped     : {s.get('final_mapped', 0)}")
            print(f"Missing          : {s.get('missing_count', 0)}")
            
            # Show workflow efficiency rates
            workflow = s.get('workflow_steps', {})
            if workflow:
                print("\nWorkflow Efficiency:")
                print(f"  Cache hit rate     : {workflow.get('cache_hit_rate', '0%')}")
                print(f"  Ref success rate   : {workflow.get('ref_success_rate', '0%')}")
                print(f"  Two-step success   : {workflow.get('two_step_success_rate', '0%')}")
                print(f"  Overall success    : {workflow.get('overall_success_rate', '0%')}")
                print(f"  Missing rate       : {workflow.get('missing_rate', '0%')}")
            
            if s.get("error"):
                print(f"Last error       : {s['error']}")

    def get_matched_data(self) -> pd.DataFrame:
        return self.df

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
            "vermilion" in row["Shipment Reference Number 1"].lower() or
            "yuzhao liu" in row["Sender Name"].lower() or
            "yuzhao liu" in row["Sender Company Name"].lower() or
            "xiaorong" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000215"

        # 4. Bondex (pickup fee allocates to TWW)
        elif row["Account Number"] in ["K5811C", "F03A44"] and \
            row["Charge_Cate_EN"] not in ["Daily Pickup", "Daily Pickup - Fuel"]:
            return "F000281"

        # 5. SPT
        elif row["Account Number"] == "HE6132":
            return "F000208"

        # 6. Transware
        elif row["Account Number"] in (["H930G2", "H930G3", "H930G4", "R1H015", "XJ3936"]):
            return "F000222"
        elif not is_blank(row["Tracking Number"]) or not is_blank(row["Lead Shipment Number"]):
            return "F000222"
        
        # 7. STARTING INTERNATIONAL
        elif "S202508220027-P" in row["Shipment Reference Number 1"].upper():
            return "F000253"

        # 8. Daily Pickup logic
        elif row["Charge_Cate_EN"] in ["Daily Pickup", "Daily Pickup - Fuel"]:
            return self.dict_pickup.get(row["Account Number"], {}).get("Cust.ID", self.DEFAULT_CUST_ID)

        # # 8. TWW
        # elif row["Sender Company Name"].lower() == "tww" or \
        #     row["Sender Company Name"].lower() == "twnj":
        #     return "F000299"
        
        # 9. Generic cost rules
        elif str(row["Charge_Cate_EN"]).upper() in ["SCC AUDIT FEE", "POD FEE"]:
            return self.DEFAULT_CUST_ID

        # Fallback
        print(f"account number: {row['Account Number']}")
        print(f"Charge_Cate_EN: {row['Charge_Cate_EN']}")
        print(f"pass all exception handlings. 'Charge Description':{row['Charge Description']}, 'Charge_Cate_EN':{row['Charge_Cate_EN']}, 'Amount':{row['Net Amount']}")
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
            "ZONE ADJUSTMENT ",
            "Returns "
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
        condition_exclude = ~df["Account Number"].isin(self.excluded_from_exception)
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
        # Conditional logic for 客户单号(文本格式)
        # Case 1: When Lead Shipment Number == Shipment Reference Number 1
        mask_same_ref = (df_exceptions["Lead Shipment Number"].fillna("") == 
                        df_exceptions["Shipment Reference Number 1"].fillna(""))
        
        # For matching rows: use only cleaned Shipment Reference Number 1
        ref_only = (
            df_exceptions["Shipment Reference Number 1"].fillna("NoRef").astype(str)
            .str.replace(" ", "", regex=False)  # Remove all spaces
            .str.replace(",", "_", regex=False)  # Replace commas with underscores
        )        
        # For non-matching rows: use original combined logic
        ref_plus_lead = (
            df_exceptions["Shipment Reference Number 1"].fillna("NoRef").astype(str)
            .str.replace(" ", "", regex=False)  # Remove all spaces
            .str.replace(",", "_", regex=False) + "-" +  # Replace commas with underscores
            df_exceptions["Lead Shipment Number"].fillna("UNKNOWN").astype(str)
        )
        
        # Apply conditional logic and truncate to 34 characters
        output_template["客户单号(文本格式)"] = ref_plus_lead.where(~mask_same_ref, ref_only).str[:34]        
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
        for idx, row in self.df.iterrows():
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
            else:
                invoice = self.invoices[inv_num]

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

            def _nn(v):
                return None if pd.isna(v) else v

            # initialize dims directly
            package.length = _nn(row["Billed Length"])
            package.width = _nn(row["Billed Width"])
            package.height = _nn(row["Billed Height"])
            package.entered_length = _nn(row["Entered Length"])
            package.entered_width = _nn(row["Entered Width"])
            package.entered_height = _nn(row["Entered Height"])

            package.pkg_ref1 = row["Package Reference Number 1"]
            package.pkg_ref2 = row["Package Reference Number 2"]

        else:
            package = shipment.packages[pkg_trk_num]
            def _nn(v):
                return None if pd.isna(v) else v
            # Only update if not blank for dims (keep previous if blank)
            if not is_blank(row["Billed Length"]):
                package.length = _nn(row["Billed Length"])
            if not is_blank(row["Billed Width"]):
                package.width = _nn(row["Billed Width"])
            if not is_blank(row["Billed Height"]):
                package.height = _nn(row["Billed Height"])
            if not is_blank(row["Entered Length"]):
                package.entered_length = _nn(row["Entered Length"])
            if not is_blank(row["Entered Width"]):
                package.entered_width = _nn(row["Entered Width"])
            if not is_blank(row["Entered Height"]):
                package.entered_height = _nn(row["Entered Height"])

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
        "Payment Processing Fee": "6050",
        "SCC Audit Fee": "7152",
        "Daily Pickup": "7151",
        "Daily Pickup - Fuel": "7151",
        "POD Fee": "7154",
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
        self.raw_path = base / "data" / "raw_invoices" / self.batch_number

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

    # helper that blanks out missing values and formats integers nicely(explicit for Dim)
    @staticmethod
    def _fmt_inch(v):
        try:
            f = float(v)
            if np.isnan(f) or f == 0:
                return ""       # treat 0/NaN as missing
            return str(int(f)) if f.is_integer() else f"{f:.1f}"
        except Exception:
            return ""

    @staticmethod
    def _fmt_inch_triplet(L, W, H):
        a = UpsInvoiceExporter._fmt_inch(L)
        b = UpsInvoiceExporter._fmt_inch(W)
        c = UpsInvoiceExporter._fmt_inch(H)
        return f"{a}x {b}x {c}" if a and b and c else ""

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

                # Packages (and package-level charges)# Packages (and package-level charges)
                for pkg in getattr(ship, "packages", {}).values():
                    # weights
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

                    Entered_L, Entered_W, Entered_H = getattr(pkg, "entered_length", None), getattr(pkg, "entered_width", None), getattr(pkg, "entered_height", None)
                    enter_dim = self._fmt_inch_triplet(Entered_L, Entered_W, Entered_H)
                    L, W, H = getattr(pkg, "length", None), getattr(pkg, "width", None), getattr(pkg, "height", None)
                    bill_dim = self._fmt_inch_triplet(L, W, H)

                    p_append({
                        "cust_id": cust_id,
                        "Invoice Number": inv_num,
                        "Invoice Date": inv_date,
                        "Lead Shipment Number": main_trk,
                        "Tracking Number": getattr(pkg, "trk_num", "") or main_trk,
                        "Billed Weight": w_lb,
                        "Billed Weight (kg)": w_kg,
                        "Entered Dim": enter_dim,
                        "Length (cm)": to_cm(getattr(pkg, "length", "")),
                        "Width (cm)":  to_cm(getattr(pkg, "width", "")),
                        "Height (cm)": to_cm(getattr(pkg, "height", "")),
                        "Bill Dim": bill_dim, 
                        "Sender Postal Ref":  getattr(ship.sender, "zipcode", ""),
                        "Receiver Postal Ref": getattr(ship.consignee, "zipcode", ""),
                        "Zone": getattr(ship, "zone", ""),
                        "Comm/Res": "",
                        "TrsDt": getattr(ship, "tran_date", ""), 
                        "Ref1": getattr(ship, "ship_ref1", ""),
                        "Ref2": getattr(ship, "ship_ref2", ""),
                        "PkgID1": getattr(pkg, "pkg_ref1", ""),
                        "PkgID2": getattr(pkg, "pkg_ref2", ""),
                        "Sender Name":    getattr(ship.sender, "contact", ""),
                        "Sender Company Name": getattr(ship.sender, "company", ""),
                        "Sender Address Line 1": getattr(ship.sender, "addr1", ""),
                        "Sender Address Line 2": getattr(ship.sender, "addr2", ""),
                        "Sender City":    getattr(ship.sender, "city", ""),
                        "Sender State":   getattr(ship.sender, "state", ""),
                        "Sender Postal":  getattr(ship.sender, "zipcode", ""),
                        "Sender Country": getattr(ship.sender, "country", ""),
                        "Receiver Name":  getattr(ship.consignee, "contact", ""),
                        "Receiver Company": getattr(ship.consignee, "company", ""),
                        "Receiver Address Line 1": getattr(ship.consignee, "addr1", ""),
                        "Receiver Address Line 2": getattr(ship.consignee, "addr2", ""),
                        "Receiver City":  getattr(ship.consignee, "city", ""),
                        "Receiver State": getattr(ship.consignee, "state", ""),
                        "Receiver Postal": getattr(ship.consignee, "zipcode", ""),
                        "Receiver Country": getattr(ship.consignee, "country", ""),
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

        if "Bill Dim" not in self.flat_charges.columns:
            self.flat_charges["Bill Dim"] = ""            

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

        summary_invoice = df.groupby("Invoice Number")[["ap_amt"]].sum().reset_index()
        summary_customer = df.groupby("cust_id")[["ap_amt", "ar_amt"]].sum().reset_index()
        for cid in SPECIAL_CUSTOMERS:
            mask = summary_customer["cust_id"] == cid
            if mask.any():  # only update if cid exists
                factor = self.dict_ar.get(cid, {}).get("Factor", 0.0)
                summary_customer.loc[mask, "ar_amt"] = (
                    summary_customer.loc[mask, "ap_amt"] * factor
                ).round(2)

        # Calculate grand totals
        grand_ap_total = summary_invoice["ap_amt"].sum()
        grand_ar_total = summary_customer["ar_amt"].sum() # Since there is a special ar rule for SPECIAL_CUSTOMERS

        # Add Grand Total row to summary_invoice
        grand_total_invoice_row = pd.DataFrame({
            "Invoice Number": ["Grand Total"],
            "ap_amt": [f"{grand_ap_total:.2f}"]
        })
        summary_invoice_with_total = pd.concat([summary_invoice, grand_total_invoice_row], ignore_index=True)

        # Add Grand Total row to summary_customer
        grand_total_customer_row = pd.DataFrame({
            "cust_id": ["Grand Total"],
            "ap_amt": [f"{grand_ap_total:.2f}"],
            "ar_amt": [f"{grand_ar_total:.2f}"]
        })

        # Calculate YDD import totals (exclude special customers and empty cust_id which refer to general costs)
        excluded_customers = set(SPECIAL_CUSTOMERS.keys()) | {"", "F000999"}  # Add default customer ID for general costs
        ydd_summary = summary_customer[~summary_customer["cust_id"].isin(excluded_customers)]
        ydd_ap_total = ydd_summary["ap_amt"].sum()
        ydd_ar_total = ydd_summary["ar_amt"].sum()

        ydd_total_row = pd.DataFrame({
            "cust_id": ["YDD Import Total"],
            "ap_amt": [f"{ydd_ap_total:.2f}"],
            "ar_amt": [f"{ydd_ar_total:.2f}"]
        })

        summary_customer_with_totals = pd.concat([summary_customer, grand_total_customer_row, ydd_total_row], ignore_index=True)

        # Prepare general cost dataframe with formatting
        general_cost_display = self.general_cost_df[["Charge_Cate_CN", "ap_amt"]].copy()
        general_cost_display["总应付金额"] = general_cost_display["ap_amt"].apply(lambda x: f"{x:.2f}")
        general_cost_display = general_cost_display.rename(columns={"Charge_Cate_CN": "费用类型（中文）"})
        general_cost_display = general_cost_display[["费用类型（中文）", "总应付金额"]].sort_values("总应付金额", ascending=False, key=lambda x: x.astype(float))

        # Add total row for general costs
        general_cost_total = self.general_cost_df["ap_amt"].sum()
        general_total_row = pd.DataFrame({
            "费用类型（中文）": ["Total"],
            "总应付金额": [f"{general_cost_total:.2f}"]
        })
        general_cost_with_total = pd.concat([general_cost_display, general_total_row], ignore_index=True)

        output_file = self.output_path / "UPS_Invoice_Export.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.fillna("").replace("nan", "").to_excel(writer, sheet_name="Details", index=False)
            summary_invoice_with_total.to_excel(writer, sheet_name="Summary by Invoice", index=False)
            summary_customer_with_totals.to_excel(writer, sheet_name="Summary by Customer", index=False)
            general_cost_with_total.to_excel(writer, sheet_name="Summary for General Cost", index=False)
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
            output_path = self.output_path / f"{cid}_{self.batch_number}.csv"
            combined_inv.fillna("").replace("nan", "").to_csv(output_path, header=header_list, index=False)
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
                index=["Lead Shipment Number", "Tracking Number"],
                columns="Charge_Cate_CN",
                values="ar_amt",
                aggfunc="sum",
                fill_value=0
            ).reset_index()
            pivot_df["Package Total"] = pivot_df.drop(columns=["Lead Shipment Number", "Tracking Number"]).sum(axis=1)
            col_totals = pivot_df.drop(columns=["Lead Shipment Number", "Tracking Number"]).sum(axis=0).to_frame().T
            col_totals.insert(0, "Lead Shipment Number", "Grand Total")
            col_totals.insert(1, "Tracking Number", "")
            pivot_df = pd.concat([pivot_df, col_totals], axis=0)

            # 2. Ensure "运费"(transporation fee) is the first column
            cols = pivot_df.columns.tolist()
            if "运费" in cols:
                cols.remove("运费")
                cols = cols[:2] + ["运费"] + cols[2:]
            pivot_df = pivot_df[cols]
            
            # 3. Optional: reset index for export
            pivot_df = pivot_df.reset_index(drop=True)

            # Ship Info
            # Drop columns
            cols_to_drop = ["cust_id", "Invoice Number", "Invoice Date", "Length (cm)", \
                            "Width (cm)", "Height (cm)"]
            pkg_sub = pkg_sub.drop(columns=cols_to_drop, errors="ignore")

            # Write
            out = self.output_path / f"{cid}_{self.batch_number}.xlsx"
            with pd.ExcelWriter(out, engine="xlsxwriter") as w:
                cover.fillna("").replace("nan", "").to_excel(w, sheet_name="Invoice", index=False)
                ar_summary.to_excel(w, sheet_name="Charge Summary", index=False)
                pivot_df.fillna("").replace("nan", "").to_excel(w, sheet_name="Detail", index=False)
                (pkg_sub if not pkg_sub.empty else pd.DataFrame(columns=["（no packages）"])).fillna("").replace("nan", "").to_excel(
                    w, sheet_name="ShpInf", index=False
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
        
        # Generate F000222 summary if F000222 invoice was created
        self.generate_f000222_summary()
        
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
            "自动对账": "N"
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
    # F000222 Summary Generator
    # ----------------------
    def generate_f000222_summary(self):
        """
        Generate F000222 invoice summary Excel spreadsheet with:
        1. Source data from F000222_{batch_number}.csv
        2. Summary grouped by Invoice Number
        3. AP Amount (Net Amount where Charge Description != "Payment Processing Fee")
        4. Processing Fee Amount (Net Amount where Charge Description == "Payment Processing Fee")
        5. AR Amount = round(AP Amount * 1.01, 2) with total adjustment for rounding errors
        """
        # Check if F000222 CSV file exists
        f000222_csv_path = self.output_path / f"F000222_{self.batch_number}.csv"
        
        if not f000222_csv_path.exists():
            print(f"⚠️ F000222 invoice file not found: {f000222_csv_path}")
            print("   Skipping F000222 summary generation.")
            return
        
        try:
            # Read the F000222 CSV file with proper encoding handling
            tried_encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            df = None
            
            for enc in tried_encs:
                try:
                    df = pd.read_csv(f000222_csv_path, encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"⚠️ Error reading {f000222_csv_path.name} with {enc}: {e}")
                    continue
            
            if df is None:
                print(f"❌ Could not read F000222 CSV file with any encoding")
                return
            
            # Ensure required columns exist
            required_cols = ["Invoice Number", "Charge Description", "Net Amount"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns in F000222 CSV: {missing_cols}")
                return
            
            # Convert Net Amount to numeric, handling any string formatting
            df["Net Amount"] = pd.to_numeric(df["Net Amount"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            
            # Create summary by Invoice Number
            summary_data = []
            
            for invoice_num in df["Invoice Number"].unique():
                invoice_df = df[df["Invoice Number"] == invoice_num]
                
                # Calculate Invoice Amount 
                invoice_amount = invoice_df["Net Amount"].sum()

                # Calculate AP Amount (exclude Payment Processing Fee)
                ap_df = invoice_df[invoice_df["Charge Description"] != "Payment Processing Fee"]
                ap_amount = ap_df["Net Amount"].sum()
                
                # Calculate Processing Fee Amount
                fee_df = invoice_df[invoice_df["Charge Description"] == "Payment Processing Fee"]
                processing_fee = fee_df["Net Amount"].sum()
                
                # Calculate AR Amount (AP * 1.01, rounded)
                ar_amount = round(ap_amount * 1.01, 2)
                
                summary_data.append({
                    "Invoice Number": invoice_num,
                    "Invoice Amount": invoice_amount,
                    "Processing Fee Amount": processing_fee,
                    "AP Amount": ap_amount,
                    "AR Amount": ar_amount
                })
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            if summary_df.empty:
                print("⚠️ No data found for F000222 summary")
                return
            
            # Calculate totals
            total_invoice_amount = summary_df["Invoice Amount"].sum()
            total_ap = summary_df["AP Amount"].sum()
            total_processing_fee = summary_df["Processing Fee Amount"].sum()
            total_ar_target = round(total_ap * 1.01, 2)
            current_ar_total = summary_df["AR Amount"].sum()
            
            # Adjust the last invoice's AR amount for rounding errors
            if len(summary_df) > 0 and current_ar_total != total_ar_target:
                adjustment = total_ar_target - current_ar_total
                last_idx = summary_df.index[-1]
                summary_df.loc[last_idx, "AR Amount"] += adjustment
                summary_df.loc[last_idx, "AR Amount"] = round(summary_df.loc[last_idx, "AR Amount"], 2)
                
                if adjustment != 0 and FLAG_DEBUG:
                    print(f"📊 Applied rounding adjustment of ${adjustment:.2f} to last invoice")
            
            # Add totals row
            totals_row = pd.DataFrame({
                "Invoice Number": ["Total"],
                "Invoice Amount": [total_invoice_amount],
                "Processing Fee Amount": [summary_df["Processing Fee Amount"].sum()],
                "AP Amount": [summary_df["AP Amount"].sum()],
                "AR Amount": [summary_df["AR Amount"].sum()]
            })
            
            summary_with_totals = pd.concat([summary_df, totals_row], ignore_index=True)
            
            # Generate output file
            output_file = self.output_path / f"F000222_{self.batch_number}_Summary.xlsx"
            
            # Create Excel file with formatting
            with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
                # Write summary data
                summary_with_totals.to_excel(writer, sheet_name="Invoice Summary", index=False)
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets["Invoice Summary"]
                
                # Add formatting
                money_format = workbook.add_format({'num_format': '$#,##0.00'})
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D3D3D3'
                })
                total_format = workbook.add_format({
                    'bold': True,
                    'num_format': '$#,##0.00',
                    'bg_color': '#E6E6FA'
                })
                
                # Set column widths
                worksheet.set_column('A:A', 20)  # Invoice Number
                worksheet.set_column('B:E', 18)  # Amount columns
                
                # Format header row
                for col_num in range(len(summary_with_totals.columns)):
                    worksheet.write(0, col_num, summary_with_totals.columns[col_num], header_format)
                
                # Format data rows with money format (excluding header and total row)
                for row_num in range(1, len(summary_with_totals)):
                    for col_num in range(1, len(summary_with_totals.columns)):  # Skip Invoice Number column
                        cell_value = summary_with_totals.iloc[row_num-1, col_num]
                        if row_num < len(summary_with_totals):  # Data rows
                            worksheet.write(row_num, col_num, cell_value, money_format)
                
                # Format total row (last row)
                total_row_num = len(summary_with_totals)
                for col_num in range(1, len(summary_with_totals.columns)):  # Skip Invoice Number column
                    cell_value = summary_with_totals.iloc[-1, col_num]
                    worksheet.write(total_row_num, col_num, cell_value, total_format)
            
            print(f"✅ F000222 invoice summary generated with {len(summary_df)} invoices")
            if FLAG_DEBUG:                
                print(f"📁 F000222 Summary saved to: {output_file}")
                print(f"💰 Total AP: ${total_ap:.2f}, Total AR: ${summary_df['AR Amount'].sum():.2f}")
            
        except Exception as e:
            print(f"❌ Error generating F000222 summary: {e}")
            import traceback
            traceback.print_exc()

    # ----------------------
    # Combined runner
    # ----------------------
    def generate_xero_templates(self):
        self.generate_xero_ap_template()
        self.generate_xero_ar_template()