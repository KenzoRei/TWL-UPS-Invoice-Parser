from pathlib import Path
import sys, pandas as pd, traceback
from ups_invoice_parser import (
    UpsInvLoader, UpsInvNormalizer, UpsCustomerMatcher,
    UpsInvoiceBuilder, UpsInvoiceExporter,
)
from contextlib import contextmanager
from time import perf_counter

out_path = Path(__file__).resolve().parent / "output"
out_path.parent.mkdir(parents=True, exist_ok=True)
FLAG_API_MATCHUP = True
CNT_THREAD = 3

class StepTimer:
    def __init__(self):
        self.durations = {}   # {label: seconds}
        self._stack = []      # for nested timing if needed

    @contextmanager
    def timeit(self, label: str):
        start = perf_counter()
        self._stack.append(label)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.durations[label] = self.durations.get(label, 0.0) + elapsed
            self._stack.pop()

    def print_summary(self, title: str = "⏱️ Runtime summary"):
        width = max((len(k) for k in self.durations), default=10)
        print("\n" + title)
        print("-" * (width + 14))
        for k, v in self.durations.items():
            print(f"{k.ljust(width)} : {v:8.3f}s")

def main():
    t = StepTimer()

    # === 1) Select + validate + archive ===
    with t.timeit("1) import (select/validate/archive)"):
        loader = UpsInvLoader()
        loader.run_import(interactive=True, cli_fallback=False)  # strict by design
        file_list = loader.invoices
        print(f"📥 Selected {len(file_list)} CSV file(s)")

    # === 2) Normalize invoices ===
    with t.timeit("2) normalize"):
        normalizer = UpsInvNormalizer(file_list)
        normalizer.load_invoices()
        normalizer.merge_invoices()
        normalizer.standardize_invoices()
        normalized_df = normalizer.get_normalized_data()
        # normalized_df.to_excel(out_path / "Normalized_Invoices.xlsx", index=False)
        print(f"✅ Normalized {len(normalized_df)} rows from {len(file_list)} files")

    # === 3) Match customers & classify charges ===
    with t.timeit("3) match & classify"):
        if FLAG_API_MATCHUP:
            matcher = UpsCustomerMatcher(normalized_df, use_api=True, ydd_threads=CNT_THREAD)
            matcher.match_customers()
            if CNT_THREAD>1:
                matched_df = matcher.get_matched_data()
        else:
            matcher = UpsCustomerMatcher(normalized_df)
            # Let the user choose exactly one 数据列表*.xlsx
            mapping_path = matcher.choose_mapping_file_dialog()
            if not mapping_path:
                raise RuntimeError("No mapping file selected.")
            matcher.match_customers()
            matched_df = matcher.get_matched_data()
        # matched_df.to_excel(out_path / "Matched_Invoices.xlsx", index=False)
        # print(f"mapping_pickup: {matcher.dict_pickup}")
        print(f"✅ Matching complete — {matched_df['cust_id'].nunique()} unique customers")

        # Better unmatched check (NaN or "")
        unassigned_mask = matched_df["cust_id"].isna() | (matched_df["cust_id"].astype(str).str.strip() == "")
        if unassigned_mask.any():
            print(f"⚠️ {unassigned_mask.sum()} rows still have blank/NaN cust_id")

    # === 4) Build composite invoice structure ===
    with t.timeit("4) build invoices + SCC"):
        builder = UpsInvoiceBuilder(matched_df)
        builder.build_invoices()
        builder._scc_handler()  # redistribute SCC fee
        invoices_dict = builder.get_invoices()
        if not invoices_dict:
            raise RuntimeError("No Invoice objects were built — check earlier steps.")
        print(f"✅ Built {len(invoices_dict)} Invoice objects")

    # === 5) Save invoices (.pkl) ===
    with t.timeit("5) save invoices (.pkl)"):
        builder.save_invoices()

    # === 6) Reload from .pkl ===
    with t.timeit("6) reload invoices (.pkl)"):
        first_invoice = next(iter(invoices_dict.values()))
        batch_number = getattr(first_invoice, "batch_num", None) or getattr(loader, "batch_number", None)
        if not batch_number:
            raise RuntimeError("Batch number not available (from invoice or loader).")
        reload_builder = UpsInvoiceBuilder(pd.DataFrame())  # empty init is fine for reload
        reload_builder.load_invoices(batch_number)
        print(f"✅ Reloaded {len(reload_builder.invoices)} invoices from saved file")

    # === 7) Initialize exporter ===
    with t.timeit("7) init exporter"):
        exporter = UpsInvoiceExporter(invoices=reload_builder.invoices)

    # === 8) Master export (Details + Summaries + General Cost) ===
    with t.timeit("8) master export"):
        exporter.export()

    # === 9) YiDiDa templates (AP + AR) ===
    with t.timeit("9) YDD AP"):
        exporter.generate_ydd_ap_template()
    with t.timeit("10) YDD AR"):
        exporter.generate_ydd_ar_template()

    # === 10) Xero templates (AP + AR) ===
    with t.timeit("11) Xero templates"):
        exporter.generate_xero_templates()

    # === 11) Per-customer workbooks ===
    with t.timeit("12) per-customer workbooks"):
        exporter.generate_customer_invoices()

    print(f"✅ All exports completed for batch {batch_number}")
    print(f"📁 Output folder: {(Path(__file__).resolve().parent / 'output' / batch_number)}")

    # === Summary ===
    t.print_summary()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
