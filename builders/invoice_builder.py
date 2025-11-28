"""UPS Invoice Builder for constructing Invoice/Shipment/Package/Charge objects."""

import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

from config import SCC_UNIT_CHARGE
from models import Invoice, Shipment, Package, Charge, Location
from utils.helpers import is_blank


class UpsInvoiceBuilder:
    """Build nested Invoice → Shipment → Package → Charge structure."""
    
    def __init__(self, normalized_df: pd.DataFrame):
        """
        Initialize builder.
        
        Args:
            normalized_df: Normalized and matched DataFrame with customer info
        """
        self.df = normalized_df
        self.output_path = Path(__file__).resolve().parent.parent / "data" / "raw_invoices"
        self.invoices: Dict[str, Invoice] = {}
        
        # Dict to store SCC packages where shipment_trk_num and inv_num were kept in tuple
        self.scc_packages: Dict[str, tuple[str, str]] = {}
        self.scc_unit_charge = SCC_UNIT_CHARGE

    def _parse_date(self, val):
        """
        Parse date value safely.
        
        Args:
            val: Date value to parse
            
        Returns:
            date object or None
        """
        if pd.isna(val):
            return None
        return pd.to_datetime(val).date()

    def _verify_invoice(self, df: pd.DataFrame) -> list:
        """
        Verify required columns exist in DataFrame.
        
        Args:
            df: DataFrame to verify
            
        Returns:
            List of missing column names
        """
        missing_cols = []
        col_names = [
            "Account Number", "Invoice Date", "Invoice Number",
            "Invoice Currency Code", "Invoice Amount",
            "Transaction Date", "Lead Shipment Number",
            "Shipment Reference Number 1", "Shipment Reference Number 2",
            "Tracking Number", "Package Reference Number 1",
            "Package Reference Number 2", "Entered Weight",
            "Billed Weight", "Billed Weight Type", "Billed Length",
            "Billed Width", "Billed Height", "Zone", "Charge_Cate_EN",
            "Charge_Cate_CN", "Charged Unit Quantity",
            "Transaction Currency Code", "Basis Amount",
            "Incentive Amount", "Net Amount", "Sender Name",
            "Sender Company Name", "Sender Address Line 1",
            "Sender Address Line 2", "Sender City", "Sender State",
            "Sender Postal", "Sender Country", "Receiver Name",
            "Receiver Company Name", "Receiver Address Line 1",
            "Receiver Address Line 2", "Receiver City", "Receiver State",
            "Receiver Postal", "Receiver Country", "Miscellaneous Line 1",
            "Miscellaneous Line 2", "Miscellaneous Line 3",
            "Miscellaneous Line 4", "Miscellaneous Line 5",
            "Original Shipment Package Quantity", "Entered Length",
            "Entered Width", "Entered Height", "cust_id", "AR_Amount",
        ]
        for col_name in col_names:
            if col_name not in df.columns:
                missing_cols.append(col_name)
        return missing_cols

    def build_invoices(self):
        """
        Convert normalized DataFrame into nested Invoice → Shipment → Package → Charge structure.
        
        Creates Invoice objects with nested Shipments, Packages, and Charges
        based on the normalized data.
        """
        # Verify headers
        missing_cols = self._verify_invoice(self.df)
        if missing_cols != []:
            missing_cols_list = ",".join(missing_cols)
            logging.warning("Missing columns: %s", missing_cols_list)

        # Grouping & object creation logic
        for _, row in self.df.iterrows():
            inv_num = row["Invoice Number"]
            if inv_num not in self.invoices:
                # Create invoice info
                invoice = Invoice()
                invoice.carrier = "UPS"
                invoice.inv_date = self._parse_date(row["Invoice Date"])
                invoice.inv_num = row["Invoice Number"]
                invoice.acct_num = row["Account Number"]
                invoice.batch_num = invoice.inv_num[-3:]
                self.invoices[inv_num] = invoice

            invoice = self.invoices[inv_num]

            # For general invoice cost (no lead shipment)
            if is_blank(row["Lead Shipment Number"]):
                self._build_invoice_cost(row, invoice)
            # Add/update shipment info
            else:
                self._build_shipment(row, invoice)

    def _build_shipment(self, row: pd.Series, invoice: Invoice):
        """
        Build or update shipment information.
        
        Args:
            row: DataFrame row with shipment data
            invoice: Parent Invoice object
        """
        Lead_Shipment_Num = row["Lead Shipment Number"]
        if Lead_Shipment_Num not in invoice.shipments:
            # Create shipment info
            invoice.shipments[Lead_Shipment_Num] = Shipment()
            shipment = invoice.shipments[Lead_Shipment_Num]
            shipment.main_trk_num = Lead_Shipment_Num
            shipment.cust_id = row["cust_id"]
            shipment.tran_date = self._parse_date(row["Transaction Date"])
            shipment.zone = row["Zone"]
            shipment.ship_ref1 = row["Shipment Reference Number 1"]
            shipment.ship_ref2 = row["Shipment Reference Number 2"]
            self._build_location(row, shipment)
        else:
            shipment = invoice.shipments[Lead_Shipment_Num]

        if is_blank(row["Tracking Number"]):
            self._build_shipment_cost(row, shipment, invoice)
        # Add/update package info
        else:
            self._build_package(row, shipment, invoice)

    def _build_package(self, row: pd.Series, shipment: Shipment, invoice: Invoice):
        """
        Build or update package information.
        
        Args:
            row: DataFrame row with package data
            shipment: Parent Shipment object
            invoice: Parent Invoice object
        """
        pkg_trk_num = row["Tracking Number"]
        if pkg_trk_num not in shipment.packages:
            shipment.packages[pkg_trk_num] = Package()
            package = shipment.packages[pkg_trk_num]
            package.trk_num = pkg_trk_num

            package.entered_wgt = row["Entered Weight"]
            package.billed_wgt = row["Billed Weight"]
            
            # Add same weight to shipment level
            shipment.entered_wgt += row["Entered Weight"]
            shipment.billed_wgt += row["Billed Weight"]

            def _nn(v):
                """Return None if NaN, otherwise value."""
                return None if pd.isna(v) else v

            package.length = _nn(row["Billed Length"])
            package.width = _nn(row["Billed Width"])
            package.height = _nn(row["Billed Height"])

            package.pkg_ref1 = row["Package Reference Number 1"]
            package.pkg_ref2 = row["Package Reference Number 2"]
        else:
            package = shipment.packages[pkg_trk_num]

        # Update charge at package level
        self._build_package_charge(row, package, shipment, invoice)

        # Update SCC flag at package level
        # SCC rule:
        # 1. Package dimensions not empty
        # 2. Charge Category Detail Code is "SCC"
        if (
            not is_blank(row["Package Dimensions"])
            and row["Charge Category Detail Code"] == "SCC"
        ):
            package.flag_UPS_SCC = True
            self.scc_packages[package.trk_num] = (shipment.main_trk_num, invoice.inv_num)

    def _build_invoice_cost(self, row: pd.Series, invoice: Invoice):
        """
        Build invoice-level cost (charges without shipment).
        
        Args:
            row: DataFrame row with charge data
            invoice: Parent Invoice object
        """
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        
        if charge_cate not in invoice.inv_charge:
            invoice.inv_charge[charge_cate] = Charge()
            invoice_charge_detail = invoice.inv_charge[charge_cate]
            invoice_charge_detail.charge_en = row["Charge_Cate_EN"]
            invoice_charge_detail.charge_cn = row["Charge_Cate_CN"]
            invoice_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            invoice_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        
        invoice_charge_detail = invoice.inv_charge[charge_cate]
        invoice_charge_detail.ap_amt += ap_amt
        invoice_charge_detail.inc_amt += inc_amt
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt

    def _build_shipment_cost(self, row: pd.Series, shipment: Shipment, invoice: Invoice):
        """
        Build shipment-level cost (charges without package).
        
        Args:
            row: DataFrame row with charge data
            shipment: Parent Shipment object
            invoice: Parent Invoice object
        """
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        
        if charge_cate not in shipment.shipment_charge:
            shipment.shipment_charge[charge_cate] = Charge()
            shipment_charge_detail = shipment.shipment_charge[charge_cate]
            shipment_charge_detail.charge_en = row["Charge_Cate_EN"]
            shipment_charge_detail.charge_cn = row["Charge_Cate_CN"]
            shipment_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            shipment_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        
        # Update shipment general charge
        shipment_charge_detail = shipment.shipment_charge[charge_cate]
        shipment_charge_detail.ap_amt += ap_amt
        shipment_charge_detail.ar_amt += ar_amt
        shipment_charge_detail.inc_amt += inc_amt
        
        # Amount aggregation at shipment level
        shipment.ap_amt += ap_amt
        shipment.ar_amt += ar_amt
        
        # Amount aggregation at invoice level
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt

    def _build_package_charge(
        self, row: pd.Series, package: Package, shipment: Shipment, invoice: Invoice
    ):
        """
        Build package-level charge.
        
        Args:
            row: DataFrame row with charge data
            package: Parent Package object
            shipment: Parent Shipment object
            invoice: Parent Invoice object
        """
        charge_cate = row["Charge_Cate_EN"]
        ap_amt = row["Net Amount"]
        ar_amt = row["AR_Amount"]
        inc_amt = row["Incentive Amount"]
        
        if charge_cate not in package.charge_detail:
            package.charge_detail[charge_cate] = Charge()
            package_charge_detail = package.charge_detail[charge_cate]
            package_charge_detail.charge_en = row["Charge_Cate_EN"]
            package_charge_detail.charge_cn = row["Charge_Cate_CN"]
            package_charge_detail.charge_ref1 = row["Miscellaneous Line 1"]
            package_charge_detail.charge_ref2 = row["Miscellaneous Line 2"]
        
        # Update package charge
        package_charge_detail = package.charge_detail[charge_cate]
        package_charge_detail.ap_amt += ap_amt
        package_charge_detail.ar_amt += ar_amt
        package_charge_detail.inc_amt += inc_amt
        
        # Amount aggregation at shipment level
        shipment.ap_amt += ap_amt
        shipment.ar_amt += ar_amt
        
        # Amount aggregation at invoice level
        invoice.ap_amt += ap_amt
        invoice.ar_amt += ar_amt

    def _build_location(self, row: pd.Series, shipment: Shipment):
        """
        Build sender and consignee location information.
        
        Args:
            row: DataFrame row with location data
            shipment: Parent Shipment object
        """
        # Check if sender address info is empty
        if is_blank(shipment.sender.zipcode):
            addr_sender = shipment.sender
            addr_sender.company = row["Sender Company Name"]
            addr_sender.contact = row["Sender Name"]
            addr_sender.addr1 = row["Sender Address Line 1"]
            addr_sender.addr2 = row["Sender Address Line 2"]
            addr_sender.city = row["Sender City"]
            addr_sender.state = row["Sender State"]
            addr_sender.zipcode = row["Sender Postal"]
            addr_sender.country = row["Sender Country"]

        if is_blank(shipment.consignee.zipcode):
            addr_consignee = shipment.consignee
            addr_consignee.company = row["Receiver Company Name"]
            addr_consignee.contact = row["Receiver Name"]
            addr_consignee.addr1 = row["Receiver Address Line 1"]
            addr_consignee.addr2 = row["Receiver Address Line 2"]
            addr_consignee.city = row["Receiver City"]
            addr_consignee.state = row["Receiver State"]
            addr_consignee.zipcode = row["Receiver Postal"]
            addr_consignee.country = row["Receiver Country"]

    def _scc_handler(self):
        """
        Calculate shipment charge correction fee by SCC flag.
        
        Distributes SCC Audit Fee from invoice level to package level
        for packages flagged with UPS SCC.
        """
        for pkg_num, (ship_num, inv_num) in self.scc_packages.items():
            invoice = self.invoices[inv_num]
            shipment = invoice.shipments[ship_num]
            package = shipment.packages[pkg_num]

            # Check if invoice has SCC Audit Fee and it's positive
            if (
                "SCC Audit Fee" in invoice.inv_charge
                and invoice.inv_charge["SCC Audit Fee"].ap_amt > 0
            ):
                # Deduct from invoice-level fee
                invoice.inv_charge["SCC Audit Fee"].ap_amt -= self.scc_unit_charge

                # Ensure package-level SCC Audit Fee exists
                if "SCC Audit Fee" not in package.charge_detail:
                    package.charge_detail["SCC Audit Fee"] = Charge()
                    pkg_scc_charge = package.charge_detail["SCC Audit Fee"]
                    pkg_scc_charge.charge_en = "SCC Audit Fee"
                    pkg_scc_charge.charge_cn = "UPS SCC审计费"
                else:
                    pkg_scc_charge = package.charge_detail["SCC Audit Fee"]

                # Add the SCC fee at package level
                pkg_scc_charge.ap_amt += self.scc_unit_charge
                pkg_scc_charge.ar_amt += self.scc_unit_charge

                # Adjust SCC fee at shipment level
                shipment.ap_amt += self.scc_unit_charge
                shipment.ar_amt += self.scc_unit_charge

    def save_invoices(self):
        """
        Save the composite invoice structure to a .pkl file.
        
        Saves to: data/raw_invoices/<batch_number>/invoices_<batch_number>.pkl
        """
        if not self.invoices:
            print("❌ No invoices to save. Please run build_invoices() first.")
            return

        # Get batch number from any invoice
        first_invoice = next(iter(self.invoices.values()))
        batch_number = getattr(first_invoice, "batch_num", "unknown_batch")

        # Prepare folder
        batch_folder = self.output_path / batch_number
        batch_folder.mkdir(parents=True, exist_ok=True)

        # File path
        file_path = batch_folder / f"invoices_{batch_number}.pkl"

        # Save pickle
        with open(file_path, "wb") as f:
            pickle.dump(self.invoices, f)

        print(f"📁 Invoices saved to {file_path}")

    def load_invoices(self, batch_number: str):
        """
        Load previously saved invoices from a given batch number folder.
        
        Args:
            batch_number: Batch number to load
        """
        file_path = self.output_path / batch_number / f"invoices_{batch_number}.pkl"
        if not file_path.exists():
            print(f"❌ No saved invoices found at {file_path}")
            return

        with open(file_path, "rb") as f:
            self.invoices = pickle.load(f)

        print(f"✅ Invoices loaded from {file_path}")

    def get_invoices(self) -> Dict[str, Invoice]:
        """
        Get all constructed Invoice objects.
        
        Returns:
            Dictionary of invoice_number -> Invoice object
        """
        return self.invoices
