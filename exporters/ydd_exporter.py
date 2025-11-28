"""
YiDiDa (YDD) Exporter Module
=============================

Handles generation of YiDiDa AP and AR templates.

YiDiDa is a logistics management platform that requires specific
template formats for AP (Accounts Payable) and AR (Accounts Receivable).
"""

from pathlib import Path
import pandas as pd

from config import SPECIAL_CUSTOMERS
from utils.helpers import is_blank


class YddExporter:
    """
    YiDiDa template exporter.
    
    Generates:
    - YDD_AP_Template.xlsx: Accounts Payable template
    - YDD_AR_Template.xlsx: Accounts Receivable template
    
    Both exclude special customers and include only shipments with
    valid lead shipment numbers.
    """

    def __init__(self, base_exporter):
        """
        Initialize YDD exporter.
        
        Args:
            base_exporter: Parent UpsInvoiceExporter instance
        """
        self.base = base_exporter

    def generate_ydd_ap_template(self):
        """
        Generate YiDiDa AP (Accounts Payable) template.
        
        Format:
        - 客户编号 (Customer ID)
        - 转单号 (Lead Shipment Number)
        - 费用名称 (Charge Category CN)
        - 金额 (AP Amount)
        - 代理计费重 (Billed Weight)
        
        Excludes:
        - Special customers (handled separately)
        - Charges without lead shipment numbers
        
        Output: YDD_AP_Template.xlsx
        """
        self.base._ensure_flattened()

        # Exclude special customers
        df = self.base.flat_charges.copy()
        df = df[~df["cust_id"].isin(SPECIAL_CUSTOMERS)]
        df = df[~df["Lead Shipment Number"].apply(is_blank)]

        df["ap_amt"] = pd.to_numeric(df["ap_amt"], errors="coerce").fillna(0)

        # Get billed weight per shipment from flat_packages (NOT flat_charges)
        if not self.base.flat_packages.empty and "Billed Weight (kg)" in self.base.flat_packages.columns:
            bw_per_ship = (
                self.base.flat_packages
                .dropna(subset=["Lead Shipment Number"])
                .groupby("Lead Shipment Number")["Billed Weight (kg)"]
                .max()  # Billed weight per shipment is typically the max
                .round(2)
            )
        else:
            bw_per_ship = pd.Series(dtype=float)

        # Group AP by customer + shipment + charge
        grouped = (
            df.groupby(["cust_id", "Lead Shipment Number", "Charge_Cate_CN"], as_index=False)
            .agg({"ap_amt": "sum"})
        )

        # Map billed weight into the grouped rows
        grouped["代理计费重"] = grouped["Lead Shipment Number"].map(bw_per_ship).fillna("")

        # Rename + order columns for YDD
        grouped = grouped.rename(columns={
            "cust_id": "客户编号",
            "Lead Shipment Number": "转单号",
            "Charge_Cate_CN": "费用名称",
            "ap_amt": "金额",
        })
        grouped["金额"] = grouped["金额"].round(2)
        grouped = grouped[["客户编号", "转单号", "费用名称", "金额", "代理计费重"]]

        output_file = self.base.output_path / "YDD_AP_Template.xlsx"
        grouped.to_excel(output_file, index=False)
        print(f"📁 YiDiDa AP template saved to {output_file}")

    def generate_ydd_ar_template(self):
        """
        Generate YiDiDa AR (Accounts Receivable) template.
        
        Format:
        - 主提单号/客户单号/系统单号 (Lead Shipment Number)
        - 子转单号/子系统单号 (Sub Shipment - empty)
        - 费用名 (Charge Category CN)
        - 金额 (AR Amount)
        - 币种 (Currency - USD)
        - 结算单位代码 (Customer ID)
        - 内部备注 (Internal Notes - empty)
        - 公开备注 (Public Notes - empty)
        - 计量单位 (Unit - empty)
        - 覆盖追加策略 (Append Strategy - 追加)
        - 自动对账 (Auto Reconcile - N)
        
        Excludes:
        - Special customers (handled separately)
        - Charges without lead shipment numbers
        
        Output: YDD_AR_Template.xlsx
        """
        self.base._ensure_flattened()

        # Exclude special customers
        df = self.base.flat_charges.copy()
        df = df[~df["cust_id"].isin(SPECIAL_CUSTOMERS)]
        df = df[~df["Lead Shipment Number"].apply(is_blank)]

        df["ar_amt"] = pd.to_numeric(df["ar_amt"], errors="coerce").fillna(0)

        # Group AR by shipment + charge + customer
        grouped = (
            df.groupby(["Lead Shipment Number", "Charge_Cate_CN", "cust_id"], as_index=False)
            .agg({"ar_amt": "sum"})
        )
        grouped["ar_amt"] = grouped["ar_amt"].round(2)

        # Build YDD AR template
        ar_df = pd.DataFrame({
            "主提单号/客户单号/系统单号": grouped["Lead Shipment Number"],
            "子转单号/子系统单号": "",
            "费用名": grouped["Charge_Cate_CN"],
            "金额": grouped["ar_amt"],
            "币种": "USD",
            "结算单位代码": grouped["cust_id"],
            "内部备注": "",
            "公开备注": "",
            "计量单位": "",
            "覆盖追加策略": "追加",
            "自动对账": "N"
        })

        output_file = self.base.output_path / "YDD_AR_Template.xlsx"
        ar_df.to_excel(output_file, index=False)
        print(f"📁 YiDiDa AR template saved to {output_file}")
