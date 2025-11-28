"""UPS Invoice file loader and validator."""

import csv
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from utils.file_chooser import FileChooser


class UpsInvLoader:
    """Load and validate UPS invoice CSV files."""
    
    def __init__(self, input_dir: str = "."):
        """
        Initialize loader.
        
        Args:
            input_dir: Directory to start file selection from
        """
        self.input_dir = Path(input_dir)
        self.invoices: List[Path] = []
        self.basic_info: List[dict] = []
        self.batch_number: Optional[str] = None
        self._chooser = FileChooser(self.input_dir)

    def choose_files_dialog(self) -> List[Path]:
        """
        Open GUI dialog to select CSV files.
        
        Returns:
            List of selected file paths
        """
        self.invoices = self._chooser.pick_csvs(interactive=True, cli_fallback=True)
        return self.invoices

    def choose_files_cli(self) -> List[Path]:
        """
        Select CSV files via command line interface.
        
        Returns:
            List of selected file paths
        """
        self.invoices = self._chooser.pick_csvs(interactive=False, cli_fallback=True)
        return self.invoices

    def _read_first_row(self, file: Path) -> List[str]:
        """
        Read first row from CSV with encoding fallback.
        
        Tries multiple encodings to handle various file formats.
        
        Args:
            file: Path to CSV file
            
        Returns:
            List of field values from first row
        """
        tried_encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        
        for enc in tried_encs:
            try:
                with file.open("r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    return next(reader)
            except UnicodeDecodeError:
                continue
            except StopIteration:
                return []
        
        # Fallback with replacement to avoid crash
        with file.open("r", encoding="latin1", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def validate_format(self) -> bool:
        """
        Validate all selected CSV files for required format.
        
        Checks:
        1. No header row
        2. Version == '2.1' (fatal if not)
        3. 250 columns
        4. Account number format (10 chars, starts with '0000')
        5. Invoice number format (15 chars, starts with '000000')
        6. All files share same batch number (last 3 digits of invoice number)
        7. Invoice amount is numeric
        
        Returns:
            True if all files pass validation, False otherwise
        """
        if not self.invoices:
            logging.warning("No files selected for validation.")
            return False

        all_valid = True
        batch_seen: Optional[str] = None

        for file in self.invoices:
            row = self._read_first_row(file)
            fname = file.name

            if not row:
                logging.warning(f"[{fname}] Empty file or unreadable first row.")
                all_valid = False
                continue

            # Check for header (heuristic)
            v_raw = row[0].strip() if len(row) >= 1 else ""
            try:
                float(v_raw)
            except ValueError:
                logging.warning(
                    f"[{fname}] Header detected or invalid version field '{v_raw}'. "
                    f"Files must NOT have a header."
                )
                all_valid = False

            # Version check (must be exactly '2.1')
            if v_raw != "2.1":
                logging.error(
                    f"[{fname}] Invoice output version must be '2.1', got '{v_raw}'."
                )
                all_valid = False
                continue

            # Column count
            if len(row) != 250:
                logging.warning(
                    f"[{fname}] Column count expected 250, got {len(row)}."
                )
                all_valid = False

            # Account number format
            acct = row[2].strip() if len(row) >= 3 else ""
            if not (len(acct) == 10 and acct.startswith("0000")):
                logging.warning(
                    f"[{fname}] Account number format invalid: '{acct}' "
                    f"(expect 10 chars starting with '0000')."
                )
                all_valid = False

            # Invoice number format + batch extraction
            inv = row[5].strip() if len(row) >= 6 else ""
            if not (len(inv) == 15 and inv.startswith("000000")):
                logging.warning(
                    f"[{fname}] Invoice number format invalid: '{inv}' "
                    f"(expect 15 chars starting with '000000')."
                )
                all_valid = False
            else:
                bn = inv[-3:]  # Proposed batch number
                if batch_seen is None:
                    batch_seen = bn
                elif bn != batch_seen:
                    logging.error(
                        f"[{fname}] Batch number '{bn}' differs from previous '{batch_seen}'. "
                        f"All invoices must be in the SAME batch."
                    )
                    all_valid = False

            # Invoice amount numeric check
            inv_amt_raw = row[10].strip() if len(row) >= 11 else ""
            try:
                float(inv_amt_raw.replace(",", ""))
            except ValueError:
                logging.warning(
                    f"[{fname}] Invoice amount (11th field) not numeric: '{inv_amt_raw}'."
                )
                all_valid = False

        # Only set batch_number if validation passed and we saw one
        if all_valid and batch_seen is not None:
            self.batch_number = batch_seen
        else:
            self.batch_number = None

        return all_valid

    def archive_raw_invoices(self) -> None:
        """
        Archive selected CSV files to data/raw_invoices/<batch_number>.
        
        Copies all selected files to the archive directory, creating
        subdirectories as needed.
        
        Raises:
            ValueError: If no files are selected
        """
        if not self.invoices:
            raise ValueError("No files selected to archive. Run choose_files_dialog() first.")

        base_path = Path(__file__).resolve().parent.parent / "data" / "raw_invoices"
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
        Run full import workflow: choose → validate → archive.
        
        Args:
            interactive: Use GUI dialog if True, otherwise CLI
            cli_fallback: Fall back to CLI if GUI fails
            
        Raises:
            ValueError: If no files selected or validation fails
        """
        # 1) Choose files
        if interactive:
            try:
                self.choose_files_dialog()
            except Exception:
                if not cli_fallback:
                    raise
                self.choose_files_cli()
        
        if not self.invoices:
            raise ValueError("No CSV files selected.")

        # 2) Validate
        if not self.validate_format():
            raise ValueError("Validation failed; see logs.")

        # 3) Archive
        self.archive_raw_invoices()
