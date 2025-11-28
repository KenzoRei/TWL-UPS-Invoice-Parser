"""
Xero Exporter Module
====================

Handles generation of Xero AP and AR templates.

Xero is an accounting software that requires specific CSV formats
for importing invoices.
"""

from pathlib import Path
from datetime import timedelta
import pandas as pd


class XeroExporter:
    """
    Xero template exporter.
    
    Generates:
    - Xero_AP_Template.csv: Accounts Payable template
    - Xero_AR_Template.csv: Accounts Receivable template
    
    Both use Xero contact and inventory item mappings.
    """

    def __init__(self, base_exporter):
        """
        Initialize Xero exporter.
        
        Args:
            base_exporter: Parent UpsInvoiceExporter instance
        """
        self.base = base_exporter

    def generate_xero_ap_template(self):
        """
        Generate Xero AP (Accounts Payable) template.
        
        Format (CSV columns):
        - *ContactName: UPS (supplier)
        - *InvoiceNumber: Batch number
        - *InvoiceDate: Invoice date
        - *DueDate: Invoice date + 30 days
        - InventoryItemCode: Item code
        - Description: Charge category or item name
        - *Quantity: 1
        - *UnitAmount: AP amount
        - *AccountCode: Purchase account code
        - *TaxType: Tax Exempt
        
        Combines:
        - General costs (no customer)
        - Customer-specific costs
        
        Output: Xero_AP_Template.csv
        """
        self.base._split_costs()
        combined_df = pd.concat(
            [self.base.general_cost_df, self.base.customer_cost_df],
            ignore_index=True,
            sort=False
        )

        # Common fields
        combined_df["*ContactName"] = "UPS"
        combined_df["*InvoiceNumber"] = self.base.batch_number
        combined_df["*InvoiceDate"] = self.base.inv_date
        combined_df["*DueDate"] = self.base.inv_date + timedelta(days=30)
        combined_df["InventoryItemCode"] = combined_df["*ItemCode"]

        # Description: charge category for general, item name for customer
        combined_df.loc[combined_df["SourceType"] == "general", "Description"] = (
            combined_df["Charge_Cate_EN"]
        )
        combined_df.loc[combined_df["SourceType"] == "customer", "Description"] = (
            combined_df["*ItemCode"].map(
                lambda code: self.base.dict_inventory.get(code, {}).get("ItemName", "UPS Services")
            )
        )

        # Amount fields
        combined_df["*Quantity"] = 1
        combined_df["*UnitAmount"] = combined_df["ap_amt"]
        combined_df["*TaxType"] = "Tax Exempt"

        # Output columns
        final_cols = [
            "*ContactName", "*InvoiceNumber", "*InvoiceDate", "*DueDate",
            "InventoryItemCode", "Description", "*Quantity", "*UnitAmount",
            "*AccountCode", "*TaxType"
        ]
        output_file = self.base.output_path / "Xero_AP_Template.csv"
        combined_df[final_cols].to_csv(output_file, index=False)
        print(f"📁 Xero AP template saved to {output_file}")

    def generate_xero_ar_template(self):
        """
        Generate Xero AR (Accounts Receivable) template.
        
        Format (CSV columns):
        - *ContactName: Customer name from mapping
        - *InvoiceNumber: Customer ID + batch number
        - *InvoiceDate: Today's date
        - *DueDate: Today + 30 days
        - InventoryItemCode: Item code
        - *Description: UPS Services
        - *Quantity: 1
        - *UnitAmount: AR amount
        - *AccountCode: Sales account code
        - *TaxType: Tax Exempt
        
        Uses:
        - customer_cost_df from split_costs
        - Xero contact mapping for customer names
        - Xero inventory mapping for account codes
        
        Output: Xero_AR_Template.csv
        
        Note: Must call generate_xero_ap_template() first to populate customer_cost_df.
        """
        if not hasattr(self.base, "customer_cost_df") or self.base.customer_cost_df.empty:
            raise ValueError("Run generate_xero_ap_template() first to populate customer_cost_df.")

        df = self.base.customer_cost_df.copy()

        # Map customer names
        df["*ContactName"] = df["cust_id"].map(
            lambda cid: self.base.dict_contacts.get(cid, {}).get("ContactName", "")
        )

        # Invoice details
        today = pd.Timestamp.today().normalize()
        df["*InvoiceNumber"] = df["cust_id"].astype(str) + "-" + self.base.batch_number
        df["*InvoiceDate"] = today
        df["*DueDate"] = today + timedelta(days=30)
        df["InventoryItemCode"] = df["*ItemCode"]
        df["*Description"] = "UPS Services"
        df["*Quantity"] = 1
        df["*UnitAmount"] = df["ar_amt"]

        # Account code from inventory mapping
        df["*AccountCode"] = df["*ItemCode"].map(
            lambda code: self.base.dict_inventory.get(code, {}).get("SalesAccount", "")
        )
        df["*TaxType"] = "Tax Exempt"

        # Output columns
        final_cols = [
            "*ContactName", "*InvoiceNumber", "*InvoiceDate", "*DueDate",
            "InventoryItemCode", "*Description", "*Quantity", "*UnitAmount",
            "*AccountCode", "*TaxType"
        ]
        output_file = self.base.output_path / "Xero_AR_Template.csv"
        df[final_cols].to_csv(output_file, index=False)
        print(f"📁 Xero AR template saved to {output_file}")

    def generate_xero_templates(self):
        """Generate both Xero AP and AR templates in correct order."""
        self.generate_xero_ap_template()
        self.generate_xero_ar_template()
