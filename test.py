from pathlib import Path
import pandas as pd
from ups_invoice_parser import UpsInvNormalizer, UpsCustomerMatcher, UpsInvoiceBuilder
from ups_invoice_parser import UpsInvoiceExporter  # refactored exporter with YDD & Xero

# === 1. Setup paths ===
base_path = Path(__file__).resolve().parent
invoice_folder = base_path / "data/raw_invoices"
file_list = list(invoice_folder.glob("*.csv"))

# === 2. Normalize invoices ===
normalizer = UpsInvNormalizer(file_list)
normalizer.load_invoices()
normalizer.merge_invoices()
normalizer.standardize_invoices()
normalized_df = normalizer.get_normalized_data()
print(f"✅ Normalized {len(normalized_df)} rows from {len(file_list)} invoice files")

# === 3. Match customers & classify charges ===
matcher = UpsCustomerMatcher(normalized_df)
matcher.match_customers()
matched_df = matcher.get_matched_data()
print(f"✅ Matching complete — {matched_df['cust_id'].nunique()} unique customers found")

# === 4. Build composite invoice structure ===
builder = UpsInvoiceBuilder(matched_df)
builder.build_invoices()
builder._scc_handler()  # handle SCC fee allocation
invoices_dict = builder.get_invoices()
print(f"✅ Built {len(invoices_dict)} Invoice objects")

# === 5. Save invoices (.pkl) ===
builder.save_invoices()

# === 6. Reload from .pkl to confirm ===
first_invoice = next(iter(invoices_dict.values()))
batch_number = first_invoice.batch_num
reload_builder = UpsInvoiceBuilder(pd.DataFrame())  # empty init
reload_builder.load_invoices(batch_number)
print(f"✅ Reloaded {len(reload_builder.invoices)} invoices from saved file")

# === 7. Initialize exporter ===
exporter = UpsInvoiceExporter(invoices=reload_builder.invoices)

# === 8. Run general UPS export (Details, Summaries) ===
exporter.export()

# === 9. Generate YiDiDa templates (AP + AR) ===
exporter.generate_ydd_ap_template()
exporter.generate_ydd_ar_template()

# === 10. Generate Xero templates (AP + AR) ===
exporter.generate_xero_templates()

print(f"✅ All exports completed for batch {batch_number}")
