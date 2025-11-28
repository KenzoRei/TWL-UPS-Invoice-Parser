"""
UPS Invoice Parser - Refactored Usage Example

This file demonstrates how to use the newly refactored modular structure.
"""

from pathlib import Path
import sys
import pandas as pd
import traceback

# New modular imports
from config import FLAG_API_USE, FLAG_DEBUG
from loaders import UpsInvLoader
from normalizers import UpsInvNormalizer
# TODO: Uncomment as modules are created
# from matchers import UpsCustomerMatcher
# from builders import UpsInvoiceBuilder
# from exporters import UpsInvoiceExporter


def step1_import_normalize_match():
    """
    Step 1: Import, Normalize, and Match Customers
    
    This is the first phase of the invoice processing workflow.
    """
    print("=" * 70)
    print("STEP 1: Import, Normalize, and Match Customers")
    print("=" * 70)
    
    # === 1) Select + validate + archive ===
    print("\n📥 Loading invoices...")
    loader = UpsInvLoader()
    loader.run_import(interactive=True, cli_fallback=False)
    file_list = loader.invoices
    print(f"✅ Selected {len(file_list)} CSV file(s)")

    # === 2) Normalize invoices ===
    print("\n🔄 Normalizing invoices...")
    normalizer = UpsInvNormalizer(file_list)
    normalizer.load_invoices()
    normalizer.merge_invoices()
    normalizer.standardize_invoices()
    normalized_df = normalizer.get_normalized_data()
    print(f"✅ Normalized {len(normalized_df)} rows from {len(file_list)} files")

    # === 3) Match customers & classify charges ===
    print("\n🔍 Matching customers...")
    # TODO: Uncomment when matcher module is created
    # matcher = UpsCustomerMatcher(normalized_df, use_api=FLAG_API_USE, ydd_threads=3)
    # matcher.match_customers()
    # matched_df = matcher.get_matched_data()
    # print(f"✅ Matching complete — {matched_df['cust_id'].nunique()} unique customers")
    
    # # Check for unmapped charges
    # unassigned_mask = matched_df["cust_id"].isna() | (matched_df["cust_id"].astype(str).str.strip() == "")
    # if unassigned_mask.any():
    #     print(f"⚠️  {unassigned_mask.sum()} rows still have blank/NaN cust_id")
    # else:
    #     print("✅ All charges mapped to customers")
    
    # # Save for step 2
    # matched_df.to_pickle("matched_invoices.pkl")
    # print("\n💾 Matched invoices saved to matched_invoices.pkl")
    
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)


def step2_build_export():
    """
    Step 2: Build Invoices and Export Results
    
    This is the second phase where invoices are constructed and exported.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Build Invoices and Export Results")
    print("=" * 70)
    
    # Load matched invoices from step 1
    print("\n📂 Loading matched invoices...")
    matched_df = pd.read_pickle("matched_invoices.pkl")
    print(f"✅ Loaded {len(matched_df)} rows")

    # TODO: Uncomment when builder module is created
    # # === 4) Build composite invoice structure ===
    # print("\n🏗️  Building invoices...")
    # builder = UpsInvoiceBuilder(matched_df)
    # builder.build_invoices()
    # builder._scc_handler()
    # invoices_dict = builder.get_invoices()
    # if not invoices_dict:
    #     raise RuntimeError("No Invoice objects were built — check earlier steps.")
    # print(f"✅ Built {len(invoices_dict)} Invoice objects")

    # # === 5) Save invoices (.pkl) ===
    # print("\n💾 Saving invoices...")
    # builder.save_invoices()

    # # === 6) Reload from .pkl ===
    # print("\n📂 Reloading invoices...")
    # first_invoice = next(iter(invoices_dict.values()))
    # batch_number = getattr(first_invoice, "batch_num", None)
    # if not batch_number:
    #     raise RuntimeError("Batch number not available (from invoice).")
    # reload_builder = UpsInvoiceBuilder(pd.DataFrame())
    # reload_builder.load_invoices(batch_number)
    # print(f"✅ Reloaded {len(reload_builder.invoices)} invoices from saved file")

    # # === 7) Initialize exporter ===
    # print("\n📤 Initializing exporter...")
    # exporter = UpsInvoiceExporter(invoices=reload_builder.invoices)

    # # === 8) Master export (Details + Summaries + General Cost) ===
    # print("\n📊 Generating master export...")
    # exporter.export()

    # # === 9) YiDiDa templates (AP + AR) ===
    # print("\n📋 Generating YiDiDa templates...")
    # exporter.generate_ydd_ap_template()
    # exporter.generate_ydd_ar_template()

    # # === 10) Xero templates (AP + AR) ===
    # print("\n📋 Generating Xero templates...")
    # exporter.generate_xero_templates()

    # # === 11) Per-customer workbooks ===
    # print("\n📑 Generating customer invoices...")
    # exporter.generate_customer_invoices()

    # print(f"\n✅ All exports completed for batch {batch_number}")
    # output_folder = Path.cwd() / 'output' / str(batch_number)
    # print(f"📁 Output folder: {output_folder}")
    
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)


def main():
    """Main entry point for the refactored invoice parser."""
    try:
        # Run step 1
        step1_import_normalize_match()
        
        # Uncomment to run step 2
        # step2_build_export()
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if FLAG_DEBUG:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
