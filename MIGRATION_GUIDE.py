"""
Migration Guide: Updating test.py for Refactored Structure
===========================================================

This file shows how to update your test.py (and other scripts) to use
the new modular structure.

BEFORE (Old Monolithic Import):
--------------------------------
```python
from ups_invoice_parser import (
    UpsInvLoader,
    UpsInvNormalizer,
    UpsCustomerMatcher,
    UpsInvoiceBuilder,
    UpsInvoiceExporter,
)
```

AFTER - Option 1 (Direct Module Imports):
------------------------------------------
```python
from loaders.invoice_loader import UpsInvLoader
from normalizers.invoice_normalizer import UpsInvNormalizer
from matchers.customer_matcher import UpsCustomerMatcher
from builders.invoice_builder import UpsInvoiceBuilder
from exporters.base_exporter import UpsInvoiceExporter
```

AFTER - Option 2 (Package-Level Imports - RECOMMENDED):
--------------------------------------------------------
```python
# This works because __init__.py exports all main classes
from ups_invoice_parser import (
    UpsInvLoader,
    UpsInvNormalizer,
    UpsCustomerMatcher,
    UpsInvoiceBuilder,
    UpsInvoiceExporter,
)
```

The usage code remains IDENTICAL - only the import changes!

Complete Example:
-----------------
"""

from pathlib import Path
from ups_invoice_parser import (
    UpsInvLoader,
    UpsInvNormalizer,
    UpsCustomerMatcher,
    UpsInvoiceBuilder,
    UpsInvoiceExporter,
)


def main():
    """Complete pipeline example with new imports."""
    
    # Setup paths
    batch_number = "315"
    output_path = Path(__file__).resolve().parent / "output" / batch_number
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw invoices
    print("=" * 50)
    print("STEP 1: Loading raw invoices...")
    print("=" * 50)
    loader = UpsInvLoader()
    raw_invoices = loader.run_import()
    
    if raw_invoices is None or raw_invoices.empty:
        print("❌ No invoices loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(raw_invoices)} invoice records")
    
    # 2. Normalize data
    print("\n" + "=" * 50)
    print("STEP 2: Normalizing invoice data...")
    print("=" * 50)
    normalizer = UpsInvNormalizer(raw_invoices)
    normalized_df = normalizer.run_normalization()
    
    print(f"✅ Normalized to {len(normalized_df)} records")
    
    # 3. Match customers and classify charges
    print("\n" + "=" * 50)
    print("STEP 3: Matching customers and classifying charges...")
    print("=" * 50)
    matcher = UpsCustomerMatcher(normalized_df)
    matched_df, dict_ar = matcher.run_matching()
    
    print(f"✅ Matched {len(matched_df)} records")
    
    # 4. Build Invoice objects
    print("\n" + "=" * 50)
    print("STEP 4: Building Invoice objects...")
    print("=" * 50)
    builder = UpsInvoiceBuilder(matched_df)
    invoices = builder.build_invoices()
    
    # Optional: Save invoices for later
    builder.save_invoices(invoices, output_path, batch_number)
    print(f"✅ Built {len(invoices)} Invoice objects")
    
    # 5. Export to various formats
    print("\n" + "=" * 50)
    print("STEP 5: Exporting to various formats...")
    print("=" * 50)
    
    # Get raw path for customer invoice generation
    raw_path = Path(__file__).resolve().parent / "data" / "raw_invoices" / batch_number
    
    exporter = UpsInvoiceExporter(
        invoices=invoices,
        batch_number=batch_number,
        output_path=output_path,
        raw_path=raw_path,
        dict_ar=dict_ar,
    )
    
    # Master export with all details
    exporter.export()
    
    # Generate customer invoices (one per customer)
    exporter.generate_customer_invoices()
    
    # Generate YiDiDa templates (AP + AR)
    exporter.generate_ydd_templates()
    
    # Generate Xero templates (AP + AR)
    exporter.generate_xero_templates()
    
    print("\n" + "=" * 50)
    print("✅ All exports complete!")
    print("=" * 50)
    print(f"📁 Output directory: {output_path}")


if __name__ == "__main__":
    main()


"""
Key Changes Summary:
====================

1. Import Statement:
   OLD: from ups_invoice_parser import UpsInvLoader, ...
   NEW: from ups_invoice_parser import UpsInvLoader, ...
   (Same! Because __init__.py exports everything)

2. Alternative Direct Imports:
   from loaders.invoice_loader import UpsInvLoader
   from normalizers.invoice_normalizer import UpsInvNormalizer
   from matchers.customer_matcher import UpsCustomerMatcher
   from builders.invoice_builder import UpsInvoiceBuilder
   from exporters.base_exporter import UpsInvoiceExporter

3. Usage Code:
   NO CHANGES NEEDED! All class names and methods remain identical.

4. Benefits:
   - Cleaner code organization
   - Easier to test individual components
   - Better IDE support with smaller files
   - Easier to maintain and debug
   - Can import only what you need

Testing Checklist:
==================
[ ] Verify imports work (run: python test.py)
[ ] Check all files are created in output directory
[ ] Validate UPS_Invoice_Export.xlsx has all sheets
[ ] Check customer invoices are generated
[ ] Verify YDD templates have correct format
[ ] Verify Xero templates have correct format
[ ] Test with different batch numbers
[ ] Test error handling with missing files

For more details, see REFACTORING_GUIDE.md
"""
