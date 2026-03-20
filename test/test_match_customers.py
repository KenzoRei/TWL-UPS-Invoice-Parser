"""
Test: full customer-matching pipeline for a real UPS invoice row.

Raw invoice line (2026-03-14, account 0000GH3239):
  2.1,0000GH3239,0000GH3239,US,2026-03-14,000000GH3239116,E,,,USD,226.85,
  2026-03-12,,2909233ACD4,,,,,0,0000000,,,,,,,0.0,,0.0,,UNC,,,,ADJ,OCG,,
  ,,,,,,FSC,FSC,Fuel Surcharge,0000001,,0.00,,USD,0.66,1.54,,0.00,0.00,,
  0.00,0.000000000,0.00,0.00,0.00,2026-03-28,,,,Mevin Yang,ZWIN,919 Fairmount

Key fields (column indices from OriHeadr.csv):
  idx  2  Account Number              = "0000GH3239"
  idx 13  Lead Shipment Number        = "2909233ACD4"   ← no Tracking Number
  idx 15  Shipment Reference Number 1 = ""
  idx 20  Tracking Number             = ""
  idx 35  Charge Category Detail Code = "OCG"
  idx 43  Charge Classification Code  = "FSC"
  idx 45  Charge Description          = "Fuel Surcharge"
  idx 52  Net Amount                  = 1.54

Expected cust_id: F000302  (resolved via YDD two-step API lookup)
"""
import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ups_invoice_parser import UpsCustomerMatcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACCOUNT_NUMBER   = "0000GH3239"
LEAD_SHIPMENT    = "2909233ACD4"
EXPECTED_CUST_ID = "F000302"

# Simulates what _load_mapping_api() would put in self.trk_to_cust after the
# YDD two-step match (piece_detail → yundan_detail) resolves 2909233ACD4.
# Key is Lead Shipment Number because Tracking Number is blank in this row —
# _collect_trk_ref() falls back to Lead Shipment Number when Tracking is blank.
MOCK_TRK_TO_CUST = {
    LEAD_SHIPMENT: (EXPECTED_CUST_ID, LEAD_SHIPMENT),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_normalized_df() -> pd.DataFrame:
    """
    Minimal normalized DataFrame that matches UpsCustomerMatcher expectations.
    Columns omitted here are those the test doesn't exercise.
    """
    return pd.DataFrame([{
        # --- identity ---
        "Account Number":               ACCOUNT_NUMBER,
        "Invoice Number":               "000000GH3239116",
        # --- tracking ---
        "Lead Shipment Number":         LEAD_SHIPMENT,
        "Tracking Number":              "",      # intentionally blank
        "Shipment Reference Number 1":  "",
        "Shipment Reference Number 2":  "",
        # --- charge fields ---
        "Charge Category Detail Code":  "OCG",
        "Charge Classification Code":   "FSC",
        "Charge Description":           "Fuel Surcharge",
        "Net Amount":                   1.54,
        "Incentive Amount":             0.66,
        "Invoice Amount":               226.85,
        "Miscellaneous Line 1":         "",
        # --- sender ---
        "Sender Name":                  "Mevin Yang",
        "Sender Company Name":          "ZWIN",
        "Sender Address Line 1":        "919 Fairmount",
        "Sender Address Line 2":        "",
        "Sender City":                  "",
        "Sender State":                 "CA",
        "Sender Postal":                "90248",
        "Sender Country":               "US",
        # --- receiver (defaults) ---
        "Receiver Name":                "",
        "Receiver Company Name":        "",
        "Receiver Address Line 1":      "",
        "Receiver Address Line 2":      "",
        "Receiver City":                "",
        "Receiver State":               "CA",
        "Receiver Postal":              "90248",
        "Receiver Country":             "US",
        # --- dimensions/weight ---
        "Billed Weight":                0.0,
        "Billed Length":                0.0,
        "Billed Width":                 0.0,
        "Billed Height":                0.0,
    }])


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def matcher():
    """
    UpsCustomerMatcher pre-loaded with:
      - common CSV mappings (Charges / Pickups / ARCalculator)
      - YDD trk_to_cust injected directly (no live API call)
    """
    df = _make_normalized_df()
    m = UpsCustomerMatcher(df, use_api=True, use_cache=False)
    m._load_common_mappings()            # reads Charges.csv / Pickups.csv / ARCalculator.csv
    m.trk_to_cust = MOCK_TRK_TO_CUST.copy()
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMatchCustomers:

    @patch.object(UpsCustomerMatcher, "_template_generator")   # suppress file I/O
    def test_cust_id_via_ydd_two_step(self, _mock_tpl, matcher):
        """
        Shipment row:
          - Tracking Number is blank
          - Lead Shipment Number = '2909233ACD4'
          - YDD two-step match (piece_detail → yundan_detail) → F000302

        match_customers() must fall back to Lead Shipment Number for the
        trk_to_cust lookup when Tracking Number is blank, mirroring the
        cache-building behaviour of _collect_trk_ref().
        """
        matcher.match_customers()
        result = matcher.get_matched_data()
        assert result.iloc[0]["cust_id"] == EXPECTED_CUST_ID

    @patch.object(UpsCustomerMatcher, "_template_generator")
    def test_charge_classified_as_fuel_surcharge(self, _mock_tpl, matcher):
        """The Fuel Surcharge row should be classified correctly, not as 'Not Defined Charge'."""
        matcher.match_customers()
        result = matcher.get_matched_data()
        assert result.iloc[0]["Charge_Cate_EN"] == "Fuel Surcharge"

    @patch.object(UpsCustomerMatcher, "_template_generator")
    def test_lead_shipment_preserved(self, _mock_tpl, matcher):
        """Lead Shipment Number should remain '2909233ACD4' after matching."""
        matcher.match_customers()
        result = matcher.get_matched_data()
        assert result.iloc[0]["Lead Shipment Number"] == LEAD_SHIPMENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
