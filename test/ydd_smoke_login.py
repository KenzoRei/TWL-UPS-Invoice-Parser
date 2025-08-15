# ydd_smoke_login.py
from __future__ import annotations
import argparse, getpass
import requests
from YDD_Client import YDDClient, build_ref_to_cust

def main():
    ap = argparse.ArgumentParser(description="YDD query smoke test (by danHaos/客户单号)")
    ap.add_argument("--base", default="http://twc.itdida.com/itdida-api", help="YDD base URL")
    ap.add_argument("--user", help="username (phone/email)")
    ap.add_argument("--show-token", action="store_true")
    ap.add_argument("--danhao", action="append", help="客户单号; repeat flag for multiple", default=[])
    args = ap.parse_args()

    # default sample danHaos (yours)
    if not args.danhao:
        args.danhao = [
            "ALS01129653651",
            "ALS01130031401",
            "ALS01133352057",
        ]

    username = "5055457@qq.com"
    password = "Twc11434!"

    client = YDDClient(base=args.base, username=username, password=password)

    # 1) login
    token = client.login()
    print("✅ Login success.")
    if args.show_token:
        print("Token:", token)
    else:
        print(f"Token received: length={len(token)} chars")

    # 2) query 运单详情 (by danHaos)
    print(f"🔎 Querying {len(args.danhao)} danHaos …")
    items = client.query_yundan_detail(args.danhao, batch_size=10)

    # 3) build mapping: 客户单号 → (cust_id, 转单号)
    ref2cust = build_ref_to_cust(items)

    # 4) show results
    print("\n=== Results ===")
    for ref in args.danhao:
        cust_id, transfer_no = ref2cust.get(ref, ("", ""))
        status = "FOUND" if cust_id else "MISSING"
        print(f"{ref} -> status={status}  cust_id={cust_id or '-'}  transfer_no={transfer_no or '-'}")

    # (optional) show raw count
    print(f"\nReturned items: {len(items)}")

if __name__ == "__main__":
    main()
