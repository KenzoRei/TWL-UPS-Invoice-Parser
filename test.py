from pathlib import Path
import sys
import pandas as pd
from ups_invoice_parser import (
    UpsInvNormalizer,
    UpsCustomerMatcher,
    UpsInvoiceBuilder,
    UpsInvoiceExporter,   # refactored exporter with single-pass flatten + per-customer invoices
)

def main():
    # === 1) Setup paths ===
    base_path = Path(__file__).resolve().parent
    invoice_folder = base_path / "data" / "raw_invoices"
    if not invoice_folder.exists():
        raise FileNotFoundError(f"Invoice folder not found: {invoice_folder}")

    file_list = sorted(invoice_folder.glob("*.csv"))
    if not file_list:
        print(f"⚠️ No CSV files found in {invoice_folder}. Nothing to do.")
        return

    # === 2) Normalize invoices ===
    normalizer = UpsInvNormalizer(file_list)
    normalizer.load_invoices()
    normalizer.merge_invoices()
    normalizer.standardize_invoices()
    normalized_df = normalizer.get_normalized_data()
    print(f"✅ Normalized {len(normalized_df)} rows from {len(file_list)} invoice files")

    # ✅ Save normalized data for manual inspection
    # normalized_outfile = base_path / "output" / "normalized_df.xlsx"
    # normalized_df.to_excel(normalized_outfile, index=False)
    # print(f"📁 Normalized DataFrame saved to {normalized_outfile}")

    # === 3) Match customers & classify charges ===
    matcher = UpsCustomerMatcher(normalized_df)
    matcher.match_customers()
    matched_df = matcher.get_matched_data()
    print(f"✅ Matching complete — {matched_df['cust_id'].nunique()} unique customers found")

    # (Optional) quick sanity checks
    if matched_df["cust_id"].isna().any():
        na_cnt = matched_df["cust_id"].isna().sum()
        print(f"⚠️ {na_cnt} rows still have NaN cust_id")

    # ✅ Save matched data for manual inspection
    # matched_outfile = base_path / "output" / "matched_df.xlsx"
    # matched_df.to_excel(matched_outfile, index=False)
    # print(f"📁 Matched & Classified DataFrame saved to {matched_outfile}")

    # === 4) Build composite invoice structure ===
    builder = UpsInvoiceBuilder(matched_df)
    builder.build_invoices()
    builder._scc_handler()  # SCC fee allocation (your existing step)
    invoices_dict = builder.get_invoices()
    print(f"✅ Built {len(invoices_dict)} Invoice objects")

    # === 5) Save invoices (.pkl) ===
    builder.save_invoices()

    # === 6) Reload from .pkl to confirm ===
    first_invoice = next(iter(invoices_dict.values()))
    batch_number = first_invoice.batch_num
    reload_builder = UpsInvoiceBuilder(pd.DataFrame())  # empty init is fine for reload
    reload_builder.load_invoices(batch_number)
    print(f"✅ Reloaded {len(reload_builder.invoices)} invoices from saved file")

    # === 7) Initialize exporter (refactored: single-pass flatten inside) ===
    exporter = UpsInvoiceExporter(invoices=reload_builder.invoices)

    # === 8) Master export (Details + Summaries + General Cost sheet) ===
    exporter.export()

    # === 9) Generate YiDiDa templates (AP + AR) ===
    exporter.generate_ydd_ap_template()
    exporter.generate_ydd_ar_template()

    # === 10) Generate Xero templates (AP + AR) ===
    exporter.generate_xero_templates()

    # === 11) One Excel per customer (Invoice / AR Summary / AR Lines / Packages) ===
    exporter.generate_customer_invoices()

    print(f"✅ All exports completed for batch {batch_number}")
    print(f"📁 Output folder: {(Path(__file__).resolve().parent / 'output' / batch_number)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        raise
