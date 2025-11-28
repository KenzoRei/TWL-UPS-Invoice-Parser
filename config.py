"""Configuration constants for UPS Invoice Parser."""

from typing import Dict, Set

# ============================================================================
# GENERAL COST SETTINGS
# ============================================================================

# General costs that will NEVER allocate to any customer (first priority over all rules)
GENERAL_COST_EN: Set[str] = {
    "Payment Processing Fee"
}

# ============================================================================
# SPECIAL CUSTOMER SETTINGS
# ============================================================================

# Special Customers for special handlings:
# 1) Full charge of specific accounts are assigned to these customers
# 2) Exclude from all YDD import templates
# 3) Total AR amount = total AP amount * index, instead of aggregation of all charges
# 4) Need split original invoices instead of re-arranged invoices
SPECIAL_CUSTOMERS: Dict[str, Dict[str, Set[str]]] = {
    "F000222": {"accounts": {"H930G2", "H930G3", "H930G4", "R1H015", "XJ3936", "Y209J6", "Y215B9"}},
    "F000208": {"accounts": {"HE6132"}},
}

# Reverse mapping: account -> customer
SPECIAL_ACCT_TO_CUST: Dict[str, str] = {
    acct: cust 
    for cust, v in SPECIAL_CUSTOMERS.items() 
    for acct in v["accounts"]
}

# Set of special customer IDs
SPECIAL_CUSTS: Set[str] = set(SPECIAL_CUSTOMERS.keys())

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Enable/disable API usage for customer matching
FLAG_API_USE: bool = True

# YiDiDa API credentials
YDD_USER: str = "5055457@qq.com"
YDD_PASS: str = "Twc11434!"

# ============================================================================
# DEBUG FLAGS
# ============================================================================

# Enable debug logging and output
FLAG_DEBUG: bool = False

# ============================================================================
# XERO EXPORT SETTINGS
# ============================================================================

# Mapping for general cost items to Xero item codes
GENERAL_ITEMCODE_MAP: Dict[str, str] = {
    "Payment Processing Fee": "7154",
    "SCC Audit Fee": "7152",
    "Daily Pickup": "7151",
    "Daily Pickup - Fuel": "7151"
}

# ============================================================================
# DEFAULT VALUES
# ============================================================================

# Default customer ID for unmatched charges
DEFAULT_CUST_ID: str = "F000999"

# SCC (Shipment Charge Correction) settings
SCC_UNIT_CHARGE: float = 1.65
