from ups_invoice_parser import UpsInvNormalizer  # Your existing normalizer class
from ups_invoice_parser import UpsCustomerMatcher 
from pathlib import Path

# Get normalized data
base_path = Path(__file__).resolve().parent
invoice_folder = base_path / "data/raw_invoices"
file_list = list(invoice_folder.glob("*.csv"))
normalizer = UpsInvNormalizer(file_list)
normalizer.load_invoices()
normalizer.merge_invoices()
normalizer.standardize_invoices()
normalized_df = normalizer.get_normalized_data()

# Create matcher
matcher = UpsCustomerMatcher(normalized_df)
matcher.match_customers()
matched_df = matcher.get_matched_data()

# Save or inspect
matched_df.to_excel("matched_output.xlsx", index=False)
print("✅ Matching completed and saved.")