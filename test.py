from pathlib import Path
from ups_invoice_parser import UpsInvNormalizer

# Step 1: Gather all invoice files
base_path = Path(__file__).resolve().parent
invoice_folder = base_path / "data/raw_invoices"
file_list = list(invoice_folder.glob("*.csv"))

# Step 2: Create the normalizer instance
normalizer = UpsInvNormalizer(file_list)


# Step 3: Run normalization steps
normalizer.load_invoices()
normalizer.merge_invoices()
normalizer.standardize_invoices()

# Step 4: Get and inspect the result
df = normalizer.get_normalized_data()
print(df.head())  # Print top 5 rows
print(df.columns.tolist())  # Print all column names
