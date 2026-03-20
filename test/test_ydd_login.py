"""
Test YDD Login with different request formats
"""
import requests
import json

# Configuration
YDD_BASE = "http://twc.itdida.com/itdida-api"
USERNAME = "F000289"
PASSWORD = "abc12345"

def test_original_format():
    """Test with the exact format from the provided request (with errors)"""
    print("\n" + "="*60)
    print("Testing ORIGINAL FORMAT (as provided)")
    print("="*60)
    
    # Exact URL with query parameters (as in provided request)
    url = f"{YDD_BASE}//login?password={PASSWORD}&username={USERNAME}"
    
    headers = {
        'Accept': 'text/html,application/json,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36 Hutool',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8'
        # Note: No Content-Type specified in original request
    }
    
    # JSON body (as shown in request)
    payload = {"password": PASSWORD, "username": USERNAME}
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Body (JSON): {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"\n✓ Login SUCCESS!")
                print(f"Token: {data.get('data', 'N/A')[:50]}...")
                return data.get('data')
            else:
                print(f"\n✗ Login FAILED: {data}")
        else:
            print(f"\n✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"\n✗ Exception: {type(e).__name__}: {e}")
    
    return None


def test_correct_format_form_encoded():
    """Test with correct format (form-encoded)"""
    print("\n" + "="*60)
    print("Testing CORRECT FORMAT (form-encoded)")
    print("="*60)
    
    url = f"{YDD_BASE}/login"  # Fixed: single slash
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Form data (not JSON)
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Body (form-encoded): {payload}")
    
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=20)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"\n✓ Login SUCCESS!")
                token = data.get('data', '')
                print(f"Token: {token[:50]}..." if len(token) > 50 else f"Token: {token}")
                return token
            else:
                print(f"\n✗ Login FAILED: {data}")
        else:
            print(f"\n✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"\n✗ Exception: {type(e).__name__}: {e}")
    
    return None


def test_correct_format_json():
    """Test with JSON format (if API accepts it)"""
    print("\n" + "="*60)
    print("Testing CORRECT FORMAT (JSON)")
    print("="*60)
    
    url = f"{YDD_BASE}/login"
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    payload = {
        "username": USERNAME,
        "password": PASSWORD
    }
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Body (JSON): {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"\n✓ Login SUCCESS!")
                token = data.get('data', '')
                print(f"Token: {token[:50]}..." if len(token) > 50 else f"Token: {token}")
                return token
            else:
                print(f"\n✗ Login FAILED: {data}")
        else:
            print(f"\n✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"\n✗ Exception: {type(e).__name__}: {e}")
    
    return None


if __name__ == "__main__":
    print("YDD Login Test - Comparing Different Request Formats")
    print("NOTE: Using test credentials F000289/abc12345")
    print("      Replace with actual credentials if needed")
    
    # Test 1: Original format (with errors)
    token1 = test_original_format()
    
    # Test 2: Correct format (form-encoded)
    token2 = test_correct_format_form_encoded()
    
    # Test 3: JSON format
    token3 = test_correct_format_json()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original format (with errors):  {'✓ SUCCESS' if token1 else '✗ FAILED'}")
    print(f"Correct format (form-encoded):  {'✓ SUCCESS' if token2 else '✗ FAILED'}")
    print(f"JSON format:                    {'✓ SUCCESS' if token3 else '✗ FAILED'}")
