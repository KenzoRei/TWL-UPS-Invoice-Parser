import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ups_invoice_parser import UpsCustomerMatcher
from YDD_Client import YDDClient


TRACKING_INPUT = "1ZG2G1560307397683"
TRACKING_EXPECTED_CUST = "F000269"

ORDER_INPUT = "OUMX1648-260331-8_31174018"
ORDER_EXPECTED_CUST = "F000248"


def _make_matcher() -> UpsCustomerMatcher:
    # Minimal frame to initialize matcher internals for API helper tests.
    df = pd.DataFrame(
        [
            {
                "Tracking Number": "",
                "Lead Shipment Number": "",
                "Account Number": "TEST123",
                "Shipment Reference Number 1": "",
            }
        ]
    )
    client = YDDClient()
    return UpsCustomerMatcher(df, use_api=True, ydd_client=client, use_cache=False)


def test_ydd_two_step_tracking_to_cust_id_live():
    matcher = _make_matcher()
    result = matcher._trk2cust_two_step_matching([TRACKING_INPUT])

    assert result, f"No two-step mapping returned for tracking {TRACKING_INPUT}."

    returned_cust_ids = {cust_id for cust_id, _transfer_no in result.values()}
    assert TRACKING_EXPECTED_CUST in returned_cust_ids, (
        f"Expected cust_id {TRACKING_EXPECTED_CUST} for tracking {TRACKING_INPUT}, "
        f"but got mappings: {result}"
    )


def test_ydd_ref_to_cust_id_live():
    matcher = _make_matcher()
    result = matcher._ref2cust_matching([ORDER_INPUT])

    assert result, f"No ref mapping returned for order/ref {ORDER_INPUT}."

    returned_cust_ids = {cust_id for cust_id, _transfer_no in result.values()}
    assert ORDER_EXPECTED_CUST in returned_cust_ids, (
        f"Expected cust_id {ORDER_EXPECTED_CUST} for order/ref {ORDER_INPUT}, "
        f"but got mappings: {result}"
    )
