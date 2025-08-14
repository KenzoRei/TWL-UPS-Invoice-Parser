from __future__ import annotations
from pathlib import Path
from typing import Sequence, List

class FileChooser:
    """
    Reusable file selection helper.
    - GUI dialog via tkinter (multi- or single-select)
    - CLI fallback (paste paths) when GUI isn't available
    - Optional suffix enforcement (e.g., ['.csv'])
    - Remembers last directory used
    """
    def __init__(self, initial_dir: str | Path | None = None):
        self.initial_dir = Path(initial_dir) if initial_dir else Path.cwd()

    def pick(
        self,
        *,
        title: str = "Select files",
        patterns: Sequence[str] = ("*.*",),
        allow_multiple: bool = True,
        enforce_suffixes: Sequence[str] | None = None,
        interactive: bool = True,
        cli_fallback: bool = True,
    ) -> List[Path]:
        files: List[Path] = []
        # Try GUI dialog
        if interactive:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                filetypes = [(p, p) for p in patterns] or [("All files", "*.*")]
                init = str(self.initial_dir)
                if allow_multiple:
                    raw = filedialog.askopenfilenames(title=title, initialdir=init, filetypes=filetypes)
                else:
                    single = filedialog.askopenfilename(title=title, initialdir=init, filetypes=filetypes)
                    raw = (single,) if single else ()
                root.update(); root.destroy()
                files = [Path(p) for p in raw if p]
            except Exception:
                if not cli_fallback:
                    raise
        # CLI fallback
        if not files and (not interactive or cli_fallback):
            print(f"Enter one or more file paths (comma-separated). Expected patterns: {', '.join(patterns)}")
            s = input("> ").strip()
            files = [Path(p.strip()) for p in s.split(",") if p.strip()]

        # Enforce suffixes if requested
        if enforce_suffixes:
            ok = {s.lower() if s.startswith(".") else "." + s.lower() for s in enforce_suffixes}
            files = [p for p in files if p.suffix.lower() in ok]

        # Only existing files
        files = [p for p in files if p.exists()]

        # Remember last directory
        if files:
            self.initial_dir = files[0].parent

        return files

    def pick_csvs(self, **kwargs) -> List[Path]:
        """Convenience: multi-select CSVs."""
        return self.pick(
            title="Select UPS Invoice CSV files",
            patterns=("*.csv",),
            allow_multiple=True,
            enforce_suffixes=(".csv",),
            **kwargs,
        )

    def pick_mapping_excels(self, **kwargs) -> List[Path]:
        """Convenience: multi-select 数据列表*.xlsx (or any .xlsx/.xls)."""
        return self.pick(
            title="Select mapping files (数据列表*.xlsx)",
            patterns=("数据列表*.xlsx", "*.xlsx", "*.xls"),
            allow_multiple=True,
            enforce_suffixes=(".xlsx", ".xls"),
            **kwargs,
        )