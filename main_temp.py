# pipeline.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

# ---- utilities ----
def base_path() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

META_NAME = "stage1_meta.json"

def save_stage1_artifacts(batch: str, file_list: list[Path], out_dir: Path, normalized_df: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        norm_path = out_dir / "normalized.parquet"
        normalized_df.to_parquet(norm_path, index=False)
        norm_ref = str(norm_path)
    except Exception:
        norm_path = out_dir / "normalized.pkl"
        normalized_df.to_pickle(norm_path)
        norm_ref = str(norm_path)
    (out_dir / META_NAME).write_text(json.dumps({
        "batch": batch,
        "normalized": norm_ref,
        "selected_files": [str(p) for p in file_list],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage1_artifacts(batch: str) -> tuple[pd.DataFrame, Path]:
    out_dir = base_path() / "output" / batch
    meta = json.loads((out_dir / META_NAME).read_text(encoding="utf-8"))
    norm = Path(meta["normalized"])
    if norm.suffix == ".parquet":
        df = pd.read_parquet(norm)
    elif norm.suffix == ".pkl":
        df = pd.read_pickle(norm)
    else:
        df = pd.read_csv(norm)
    return df, out_dir


# ---- stage 1 ----
def run_stage1():
    from ups_invoice_parser import UpsInvLoader, UpsInvNormalizer
    loader = UpsInvLoader()
    loader.run_import(interactive=True, cli_fallback=True)  # choose → validate → archive
    file_list = loader.invoices
    batch = loader.batch_number
    if not batch:
        raise RuntimeError("Batch number missing after validation.")

    out_dir = base_path() / "output" / batch
    out_dir.mkdir(parents=True, exist_ok=True)

    normalizer = UpsInvNormalizer(file_list)
    normalizer.load_invoices()
    normalizer.merge_invoices()
    normalizer.standardize_invoices()
    normalized_df = normalizer.get_normalized_data()

    # write normalized invoices for the user 
    processed_file = out_dir / "processed_invoice.xlsx"
    normalized_df.to_excel(processed_file, index=False)
    print(f"📝Normalized invoices saved to: {processed_file}")

    # save artifacts for Stage 2
    save_stage1_artifacts(batch, file_list, out_dir, normalized_df)
    print(f"✅ Stage 1 complete. Artifacts saved in: {out_dir}")
    print("👉 Fill cust_id in the template (save as 数据列表-*.xlsx), then run:")
    print(f"   python pipeline.py stage2 --batch {batch}")

# ---- stage 2 ----
def run_stage2(batch: str, mapping_path: str | None):
    from ups_invoice_parser import UpsCustomerMatcher, UpsInvoiceBuilder, UpsInvoiceExporter

    normalized_df, out_dir = load_stage1_artifacts(batch)

    matcher = UpsCustomerMatcher(normalized_df)
    if mapping_path:
        matcher.set_mapping_file(Path(mapping_path))
    else:
        # interactive picker for exactly one mapping file
        selected = matcher.choose_mapping_file_dialog()
        if not selected:
            raise RuntimeError("No mapping file selected.")

    matcher.match_customers()
    matched_df = matcher.get_matched_data()

    builder = UpsInvoiceBuilder(matched_df)
    builder.build_invoices()
    builder._scc_handler()
    invoices = builder.get_invoices()
    builder.save_invoices()

    exporter = UpsInvoiceExporter(invoices=invoices)
    exporter.export()
    exporter.generate_ydd_ap_template()
    exporter.generate_ydd_ar_template()
    exporter.generate_xero_templates()
    exporter.generate_customer_invoices()

    print(f"✅ Stage 2 complete. Outputs in: {out_dir}")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="TWL UPS Invoice Parser")
    sub = ap.add_subparsers(dest="cmd")
    ap.set_defaults(cmd="stage1") 

    s1 = sub.add_parser("stage1", help="Select, validate, archive, normalize; write mapping template.")
    s2 = sub.add_parser("stage2", help="Load normalized data and run the rest.")
    s2.add_argument("--batch", required=True, help="Batch number (e.g., 315)")
    s2.add_argument("--mapping", help="Path to 数据列表*.xlsx (optional; dialog if omitted)")

    args = ap.parse_args()
    if args.cmd == "stage1":
        run_stage1()
    else:
        run_stage2(args.batch, args.mapping)

if __name__ == "__main__":
    main()
