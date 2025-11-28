"""
UPS Invoice Parser
==================

A modular system for processing UPS invoices with customer matching,
charge classification, and multi-format exports.

Main Classes:
    - UpsInvLoader: Load and validate raw invoice CSV files
    - UpsInvNormalizer: Normalize and standardize invoice data
    - UpsCustomerMatcher: Match customers and classify charges
    - UpsInvoiceBuilder: Build Invoice/Shipment/Package/Charge objects
    - UpsInvoiceExporter: Export to various formats (YDD, Xero, customer invoices)

Quick Start:
    ```python
    from ups_invoice_parser import (
        UpsInvLoader,
        UpsInvNormalizer,
        UpsCustomerMatcher,
        UpsInvoiceBuilder,
        UpsInvoiceExporter,
    )
    
    # 1. Load raw invoices
    loader = UpsInvLoader()
    raw_invoices = loader.run_import()
    
    # 2. Normalize data
    normalizer = UpsInvNormalizer(raw_invoices)
    normalized_df = normalizer.run_normalization()
    
    # 3. Match customers
    matcher = UpsCustomerMatcher(normalized_df)
    matched_df, dict_ar = matcher.run_matching()
    
    # 4. Build Invoice objects
    builder = UpsInvoiceBuilder(matched_df)
    invoices = builder.build_invoices()
    
    # 5. Export
    exporter = UpsInvoiceExporter(invoices, "315", output_path)
    exporter.export()
    exporter.generate_customer_invoices()
    exporter.generate_ydd_templates()
    exporter.generate_xero_templates()
    ```

See REFACTORING_GUIDE.md for detailed documentation.
"""

# Core classes
from loaders.invoice_loader import UpsInvLoader
from normalizers.invoice_normalizer import UpsInvNormalizer
from matchers.customer_matcher import UpsCustomerMatcher
from builders.invoice_builder import UpsInvoiceBuilder
from exporters.base_exporter import UpsInvoiceExporter

# Data models
from models import Invoice, Shipment, Package, Charge

# Configuration
from config import (
    GENERAL_COST_EN,
    GENERAL_COST_CN,
    SPECIAL_CUSTOMERS,
    FLAG_API_USE,
    YDD_USER,
    YDD_PWD,
    SCC_UNIT_CHARGE,
)

# Utilities
from utils.helpers import (
    is_blank,
    extract_dims,
    fmt_inch,
    fmt_inch_triplet,
    to_cm,
    parse_date_safe,
)

__version__ = "2.0.0"
__all__ = [
    # Core processing classes
    "UpsInvLoader",
    "UpsInvNormalizer",
    "UpsCustomerMatcher",
    "UpsInvoiceBuilder",
    "UpsInvoiceExporter",
    # Data models
    "Invoice",
    "Shipment",
    "Package",
    "Charge",
    # Configuration
    "GENERAL_COST_EN",
    "GENERAL_COST_CN",
    "SPECIAL_CUSTOMERS",
    "FLAG_API_USE",
    "YDD_USER",
    "YDD_PWD",
    "SCC_UNIT_CHARGE",
    # Utilities
    "is_blank",
    "extract_dims",
    "fmt_inch",
    "fmt_inch_triplet",
    "to_cm",
    "parse_date_safe",
]
