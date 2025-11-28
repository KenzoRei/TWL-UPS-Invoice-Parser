"""
Base Exporter Module
====================

Contains the main UpsInvoiceExporter class with common functionality
for all export operations.

This module handles:
- Loading mapping files (Xero contacts, inventory items)
- Flattening invoice composite structure to DataFrames
- Cost splitting (general vs customer-specific)
- Master export of all invoice details
"""

from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta
import pandas as pd

from models import Invoice
from config import SPECIAL_CUSTOMERS
from utils.helpers import is_blank, fmt_inch, fmt_inch_triplet

# Import sub-exporters
from .ydd_exporter import YddExporter
from .xero_exporter import XeroExporter
from .customer_exporter import CustomerExporter


class UpsInvoiceExporter:
    """
    Main exporter for UPS invoice data.
    
    Handles:
    - Loading mapping files (Xero contacts, inventory items)
    - Flattening composite Invoice objects to DataFrames
    - Splitting costs into general vs customer-specific
    - Delegating to specialized exporters (YDD, Xero, Customer)
    
    Attributes:
        invoices: List of Invoice objects to export
        batch_number: Batch identifier for export files
        output_path: Directory for output files
        raw_path: Directory containing raw invoice CSV files
        inv_date: Invoice date for Xero templates
        dict_contacts: Xero contact mapping {cust_id: contact_info}
        dict_inventory: Xero inventory mapping {item_code: item_info}
        dict_ar: AR calculator mapping {cust_id: factor_info}
    """

    def __init__(
        self,
        invoices: List[Invoice],
        batch_number: str,
        output_path: Path,
        raw_path: Path = None,
        inv_date: pd.Timestamp = None,
        dict_ar: Dict[str, Any] = None,
    ):
        """
        Initialize exporter.
        
        Args:
            invoices: List of Invoice objects to export
            batch_number: Batch identifier for this export
            output_path: Directory where export files will be saved
            raw_path: Directory containing raw invoice CSV files (for customer invoices)
            inv_date: Invoice date for Xero templates (defaults to first invoice date)
            dict_ar: AR calculator mapping from matcher (defaults to empty dict)
        """
        self.invoices = invoices
        self.batch_number = batch_number
        self.output_path = Path(output_path)
        self.raw_path = Path(raw_path) if raw_path else None
        self.inv_date = inv_date or (invoices[0].invoice_date if invoices else pd.Timestamp.today())
        self.dict_ar = dict_ar or {}

        # Mapping dicts
        self.dict_contacts: Dict[str, Dict[str, str]] = {}
        self.dict_inventory: Dict[str, Dict[str, str]] = {}

        # Flattened DataFrames (lazy-loaded)
        self.flat_charges = pd.DataFrame()
        self.flat_packages = pd.DataFrame()
        self.flat_shipments = pd.DataFrame()

        # Cost split DataFrames (lazy-loaded)
        self.general_cost_df = pd.DataFrame()
        self.customer_cost_df = pd.DataFrame()

        # Load mappings
        self._load_mappings()

        # Initialize sub-exporters
        self.ydd_exporter = YddExporter(self)
        self.xero_exporter = XeroExporter(self)
        self.customer_exporter = CustomerExporter(self)

    # ----------------------
    # Mapping loaders
    # ----------------------
    def _load_mappings(self):
        """Load Xero Contacts and Inventory Items from CSV files."""
        mappings_path = Path(__file__).resolve().parent.parent / "data" / "mappings"

        # Load Xero Contacts
        contacts_file = mappings_path / "Contacts.csv"
        if contacts_file.exists():
            contacts_df = pd.read_csv(contacts_file)
            for _, row in contacts_df.iterrows():
                cust_id = str(row.get("cust_id", ""))
                if cust_id and cust_id != "nan":
                    self.dict_contacts[cust_id] = {
                        "ContactName": row.get("ContactName", ""),
                        "EmailAddress": row.get("EmailAddress", ""),
                    }

        # Load Xero Inventory Items
        inv_files = list(mappings_path.glob("InventoryItems-*.csv"))
        if inv_files:
            inv_file = sorted(inv_files)[-1]  # Use most recent
            inv_df = pd.read_csv(inv_file)
            for _, row in inv_df.iterrows():
                code = row.get("Code", "")
                if code:
                    self.dict_inventory[str(code)] = {
                        "ItemName": row.get("Name", ""),
                        "PurchasesAccount": row.get("Purchase Account Code", ""),
                        "SalesAccount": row.get("Sales Account Code", ""),
                    }

    # ----------------------
    # Helper methods
    # ----------------------
    @staticmethod
    def _fmt_inch(val: Any) -> str:
        """Format dimension value to inches string (e.g., '12.5')."""
        return fmt_inch(val)

    @staticmethod
    def _fmt_inch_triplet(l: Any, w: Any, h: Any) -> str:
        """Format dimension triplet to 'L x W x H' string."""
        return fmt_inch_triplet(l, w, h)

    # ----------------------
    # Flattening logic
    # ----------------------
    def _flatten_all_once(self):
        """
        Traverse Invoice → Shipment → Package → Charge structure
        to create three flat DataFrames:
        - flat_charges: One row per Charge
        - flat_packages: One row per Package
        - flat_shipments: One row per Shipment
        """
        charge_rows = []
        package_rows = []
        shipment_rows = []

        for inv in self.invoices:
            inv_num = inv.invoice_number
            inv_date = inv.invoice_date
            cust_id = inv.cust_id

            for ship in inv.shipments:
                lead_num = ship.lead_shipment_number
                ship_ref1 = ship.shipment_reference1
                ship_ref2 = ship.shipment_reference2
                ship_ref3 = ship.shipment_reference3

                # Shipment-level row
                shipment_rows.append({
                    "Invoice Number": inv_num,
                    "Invoice Date": inv_date,
                    "cust_id": cust_id,
                    "Lead Shipment Number": lead_num,
                    "Shipment Reference1": ship_ref1,
                    "Shipment Reference2": ship_ref2,
                    "Shipment Reference3": ship_ref3,
                })

                for pkg in ship.packages:
                    tracking = pkg.tracking_number
                    zone = pkg.zone
                    billed_wt_kg = pkg.billed_weight_kg
                    length_cm = pkg.length_cm
                    width_cm = pkg.width_cm
                    height_cm = pkg.height_cm
                    dims_str = self._fmt_inch_triplet(pkg.length_in, pkg.width_in, pkg.height_in)

                    # Package-level row
                    package_rows.append({
                        "Invoice Number": inv_num,
                        "Invoice Date": inv_date,
                        "cust_id": cust_id,
                        "Lead Shipment Number": lead_num,
                        "Tracking Number": tracking,
                        "Zone": zone,
                        "Billed Weight (kg)": billed_wt_kg,
                        "Length (cm)": length_cm,
                        "Width (cm)": width_cm,
                        "Height (cm)": height_cm,
                        "Dimensions (inch)": dims_str,
                        "Shipment Reference1": ship_ref1,
                        "Shipment Reference2": ship_ref2,
                        "Shipment Reference3": ship_ref3,
                    })

                    for chg in pkg.charges:
                        charge_rows.append({
                            "Invoice Number": inv_num,
                            "Invoice Date": inv_date,
                            "cust_id": cust_id,
                            "Lead Shipment Number": lead_num,
                            "Tracking Number": tracking,
                            "Charge_Cate_CN": chg.charge_category_cn,
                            "Charge_Cate_EN": chg.charge_category_en,
                            "ap_amt": chg.charge_amount_ap,
                            "ar_amt": chg.charge_amount_ar,
                            "Zone": zone,
                            "Billed Weight (kg)": billed_wt_kg,
                            "Length (cm)": length_cm,
                            "Width (cm)": width_cm,
                            "Height (cm)": height_cm,
                            "Dimensions (inch)": dims_str,
                            "Shipment Reference1": ship_ref1,
                            "Shipment Reference2": ship_ref2,
                            "Shipment Reference3": ship_ref3,
                        })

        self.flat_charges = pd.DataFrame(charge_rows)
        self.flat_packages = pd.DataFrame(package_rows)
        self.flat_shipments = pd.DataFrame(shipment_rows)

    def _ensure_flattened(self):
        """Lazy initialization: flatten invoices if not already done."""
        if self.flat_charges.empty and self.invoices:
            self._flatten_all_once()

    # ----------------------
    # Cost splitting
    # ----------------------
    def _split_costs(self):
        """
        Split flat_charges into:
        - general_cost_df: General costs (no customer assignment)
        - customer_cost_df: Customer-specific costs with AR calculations
        
        For SPECIAL_CUSTOMERS, applies AR factor from dict_ar.
        """
        self._ensure_flattened()
        df = self.flat_charges.copy()

        # General costs: no lead shipment number
        mask_general = df["Lead Shipment Number"].apply(is_blank)
        general_df = (
            df[mask_general]
            .groupby("Charge_Cate_CN", as_index=False)[["ap_amt", "ar_amt"]]
            .sum()
        )
        general_df["Charge_Cate_EN"] = general_df["Charge_Cate_CN"].map(
            lambda cn: df[df["Charge_Cate_CN"] == cn]["Charge_Cate_EN"].iloc[0]
            if not df[df["Charge_Cate_CN"] == cn].empty
            else ""
        )
        general_df["*ItemCode"] = ""
        general_df["*AccountCode"] = ""
        general_df["SourceType"] = "general"

        # Customer costs: has lead shipment number
        cust_df = (
            df[~mask_general]
            .groupby("cust_id", as_index=False)[["ap_amt", "ar_amt"]]
            .sum()
        )

        # Apply AR factors for special customers
        for cid in SPECIAL_CUSTOMERS:
            mask = cust_df["cust_id"] == cid
            if mask.any():
                factor = self.dict_ar.get(cid, {}).get("Factor", 0.0)
                cust_df.loc[mask, "ar_amt"] = (
                    cust_df.loc[mask, "ap_amt"] * factor
                ).round(2)

        cust_df["*ItemCode"] = cust_df["cust_id"].astype(str).str[-4:]
        cust_df["*AccountCode"] = cust_df["*ItemCode"].map(
            lambda code: self.dict_inventory.get(code, {}).get("PurchasesAccount", "")
        )
        cust_df["SourceType"] = "customer"

        self.general_cost_df = general_df
        self.customer_cost_df = cust_df

    # ----------------------
    # Master export
    # ----------------------
    def export(self):
        """
        Export all invoice details and summaries to Excel.
        
        Creates UPS_Invoice_Export.xlsx with sheets:
        - Details: All charge records
        - Summary by Invoice: Totals per invoice
        - Summary by Customer: Totals per customer (with AR factors)
        - Summary for General Cost: General costs breakdown
        """
        self._ensure_flattened()
        self._split_costs()
        df = self.flat_charges.copy()

        # Invoice summary
        summary_invoice = (
            df.groupby("Invoice Number")[["ap_amt", "ar_amt"]]
            .sum()
            .reset_index()
        )

        # Customer summary with AR factors
        summary_customer = (
            df.groupby("cust_id")[["ap_amt", "ar_amt"]]
            .sum()
            .reset_index()
        )
        for cid in SPECIAL_CUSTOMERS:
            mask = summary_customer["cust_id"] == cid
            if mask.any():
                factor = self.dict_ar.get(cid, {}).get("Factor", 0.0)
                summary_customer.loc[mask, "ar_amt"] = (
                    summary_customer.loc[mask, "ap_amt"] * factor
                ).round(2)

        # Write to Excel
        output_file = self.output_path / "UPS_Invoice_Export.xlsx"
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            df.fillna("").replace("nan", "").to_excel(
                writer, sheet_name="Details", index=False
            )
            summary_invoice.to_excel(
                writer, sheet_name="Summary by Invoice", index=False
            )
            summary_customer.to_excel(
                writer, sheet_name="Summary by Customer", index=False
            )
            (
                self.general_cost_df[["Charge_Cate_CN", "ap_amt"]]
                .rename(columns={"Charge_Cate_CN": "费用类型（中文）", "ap_amt": "总应付金额"})
                .sort_values("总应付金额", ascending=False)
                .to_excel(writer, sheet_name="Summary for General Cost", index=False)
            )

        print(f"📁 UPS invoice export saved to {output_file}")

    # ----------------------
    # Delegation methods
    # ----------------------
    def generate_customer_invoices(self):
        """
        Generate per-customer invoices (delegates to CustomerExporter).
        
        Creates one Excel file per customer with:
        - Invoice cover (customer, batch, total)
        - Charge summary
        - Detail pivot
        - Shipment info
        """
        self.customer_exporter.generate_customer_invoices()

    def generate_ydd_ap_template(self):
        """Generate YiDiDa AP template (delegates to YddExporter)."""
        self.ydd_exporter.generate_ydd_ap_template()

    def generate_ydd_ar_template(self):
        """Generate YiDiDa AR template (delegates to YddExporter)."""
        self.ydd_exporter.generate_ydd_ar_template()

    def generate_ydd_templates(self):
        """Generate both YiDiDa AP and AR templates."""
        self.generate_ydd_ap_template()
        self.generate_ydd_ar_template()

    def generate_xero_ap_template(self):
        """Generate Xero AP template (delegates to XeroExporter)."""
        self.xero_exporter.generate_xero_ap_template()

    def generate_xero_ar_template(self):
        """Generate Xero AR template (delegates to XeroExporter)."""
        self.xero_exporter.generate_xero_ar_template()

    def generate_xero_templates(self):
        """Generate both Xero AP and AR templates."""
        self.xero_exporter.generate_xero_templates()
