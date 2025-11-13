import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from YDD_Client import YDDClient, build_ref_to_cust, select_tracking
from ups_invoice_parser import UpsCustomerMatcher
import pandas as pd

# User manual inputs
input_ref = 'S202508220027-P10'
input_tracking = '1ZG2C7946812068417'

print(f"\nTesting with:")
print(f"Reference: {input_ref}")
print(f"Tracking: {input_tracking}")
print("=" * 60)

# Initialize YDD Client
client = YDDClient()
print("Logging into YDD API...")
client.login()
print("✅ Login successful")

# Test 1: query_yundan_detail (ref to cust_id)
print("\n🔍 Test 1: query_yundan_detail (ref → cust_id)")
print("-" * 40)
try:
    data = client.query_yundan_detail([input_ref])
    print(f"Raw API response: {len(data)} items")
    
    if data:
        ref_to_cust = build_ref_to_cust(data)
        print(f"Processed mapping: {ref_to_cust}")
        
        if input_ref in ref_to_cust:
            cust_id, tracking = ref_to_cust[input_ref]
            print(f"✅ Result: {input_ref} → Customer ID: {cust_id}, Tracking: {tracking}")
        else:
            print(f"❌ No mapping found for reference: {input_ref}")
    else:
        print("❌ No data returned from API")
        
except Exception as e:
    print(f"❌ Error in query_yundan_detail: {e}")

# Test 2: query_piece_detail (tracking to ref)
print("\n🔍 Test 2: query_piece_detail (tracking → ref)")
print("-" * 40)
try:
    data = client.query_piece_detail([input_tracking])
    print(f"Raw API response: {len(data)} items")
    
    if data:
        for item in data:
            trk = select_tracking(item)
            ke_hu_dan_hao = str(item.get("keHuDanHao", "")).strip()
            print(f"✅ Result: {trk} → Reference: {ke_hu_dan_hao}")
    else:
        print("❌ No data returned from API")
        
except Exception as e:
    print(f"❌ Error in query_piece_detail: {e}")

# Test 3: _ref2cust_matching (direct ref to cust_id via UpsCustomerMatcher)
print("\n🔍 Test 3: _ref2cust_matching (ref → cust_id via UpsCustomerMatcher)")
print("-" * 60)
try:
    # Create a simple DataFrame for testing
    test_df = pd.DataFrame({
        'Tracking Number': [''],
        'Lead Shipment Number': [''],
        'Account Number': ['TEST123'],
        'Shipment Reference Number 1': [''],
    })
    
    # Initialize matcher
    matcher = UpsCustomerMatcher(test_df, use_api=True, ydd_client=client)
    
    # Test the _ref2cust_matching method directly
    result = matcher._ref2cust_matching([input_ref])
    
    print(f"_ref2cust_matching result: {result}")
    
    if result:
        for ref, (cust_id, transfer_no) in result.items():
            print(f"✅ Result: {ref} → Customer ID: {cust_id}, Transfer No: {transfer_no}")
    else:
        print(f"❌ No mapping found for reference: {input_ref}")
        
except Exception as e:
    print(f"❌ Error in _ref2cust_matching: {e}")

# Test 4: _trk2cust_two_step_matching
print("\n🔍 Test 4: _trk2cust_two_step_matching (tracking → ref → cust_id)")
print("-" * 60)
try:
    # Create a simple DataFrame for testing
    test_df = pd.DataFrame({
        'Tracking Number': [input_tracking],
        'Lead Shipment Number': [''],
        'Account Number': ['TEST123'],
        'Shipment Reference Number 1': [''],
    })
    
    # Initialize matcher
    matcher = UpsCustomerMatcher(test_df, use_api=True, ydd_client=client)
    
    # Test the two-step matching method
    result = matcher._trk2cust_two_step_matching([input_tracking])
    
    print(f"Two-step matching result: {result}")
    
    if result:
        for tracking, (cust_id, transfer_no) in result.items():
            print(f"✅ Result: {tracking} → Customer ID: {cust_id}, Transfer No: {transfer_no}")
    else:
        print("❌ No mapping found through two-step matching")
        
except Exception as e:
    print(f"❌ Error in _trk2cust_two_step_matching: {e}")

print("\n" + "=" * 60)
print("🏁 All tests completed!")
print("=" * 60)
