import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from YDD_Client import YDDClient
from ups_invoice_parser import UpsCustomerMatcher
import pandas as pd

def test_ref2cust_matching():
    """
    Test the _ref2cust_matching method with reference: S202508220027-P10
    """
    # Test reference
    test_ref = 'S202508220027-P10'
    
    print(f"\n🧪 Testing _ref2cust_matching method")
    print(f"Reference: {test_ref}")
    print("=" * 60)
    
    try:
        # Initialize YDD Client
        client = YDDClient()
        print("Logging into YDD API...")
        client.login()
        print("✅ Login successful")
        
        # Create a minimal DataFrame for testing
        test_df = pd.DataFrame({
            'Tracking Number': [''],
            'Lead Shipment Number': [''],
            'Account Number': ['TEST123'],
            'Shipment Reference Number 1': [''],
        })
        
        # Initialize UpsCustomerMatcher
        matcher = UpsCustomerMatcher(test_df, use_api=True, ydd_client=client)
        
        # Test the _ref2cust_matching method
        print(f"\n🔍 Calling _ref2cust_matching with reference: {test_ref}")
        result = matcher._ref2cust_matching([test_ref])
        
        print(f"\nRaw result: {result}")
        
        if result:
            print(f"\n✅ Success! Found mapping:")
            for ref, (cust_id, transfer_no) in result.items():
                print(f"   Reference: {ref}")
                print(f"   Customer ID: {cust_id}")
                print(f"   Transfer No: {transfer_no}")
        else:
            print(f"\n❌ No mapping found for reference: {test_ref}")
            
    except Exception as e:
        print(f"\n❌ Error in _ref2cust_matching test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_ref2cust_matching()