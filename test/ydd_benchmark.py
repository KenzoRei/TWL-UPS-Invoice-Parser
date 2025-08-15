# ydd_benchmark.py
from __future__ import annotations
import argparse, time, csv, getpass, random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOTE: uses your module name as you wrote earlier
from YDD_Client import YDDClient, build_ref_to_cust

# Default settings
YDD_USER = "5055457@qq.com"
YDD_PASS =  "Twc11434!"
FLAG_CACHE = False
CNT_THREAD = 3

# ----------------------------- IO helpers -----------------------------
def load_danhaos(csv_path: Path, *, column: str | None = None) -> List[str]:
    """
    Load danHaos from CSV. If `column` specified, use that column; else use first column.
    Skips an obvious header, de-dupes, strips, drops empties.
    """
    danhaos: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return danhaos

        if column:
            df = pd.read_csv(csv_path)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in {csv_path}")
            danhaos = df[column].astype(str).str.strip().tolist()
        else:
            first_col = [str(r[0]).strip() for r in rows if r]
            if first_col and first_col[0].lower() in {"danhao", "客户单号", "order_no", "ref"}:
                danhaos = first_col[1:]
            else:
                danhaos = first_col

    # de-dupe (preserve order) & drop empties
    return [d for d in dict.fromkeys(danhaos) if d]


# ----------------------------- parallel query -----------------------------
def query_concurrent(
    client: YDDClient,
    danhaos: List[str],
    *,
    batch_size: int = 10,
    workers: int = 3,
    max_retries: int = 4,
    base_sleep: float = 0.25,
    jitter: float = 0.15,
) -> List[dict]:
    """
    Run /queryYunDanDetail in parallel:
      - splits danhaos into batches of <=10
      - each worker uses its own requests.Session (no shared session)
      - retries on 502/503/504/429 with exponential backoff + jitter
      - relogs once if a 401 happens mid-run
    """
    batches = [danhaos[i:i + batch_size] for i in range(0, len(danhaos), batch_size)]
    if not batches:
        return []

    token = client.token
    if not token:
        raise RuntimeError("Client has no token; call client.login() before query_concurrent().")

    auth_header = {"Authorization": f"Bearer {token}"}
    base_url = client.base.rstrip("/") + "/queryYunDanDetail"
    default_timeout = 20  # seconds

    def fetch_chunk(chunk: List[str]) -> List[dict]:
        params = {"danHaos": ",".join(chunk)}
        s = requests.Session()
        s.headers.update(auth_header)
        attempt = 0
        while True:
            try:
                r = s.get(base_url, params=params, timeout=default_timeout)
                # handle 401: refresh token once via main client
                if r.status_code == 401:
                    client.login()
                    s.headers["Authorization"] = f"Bearer {client.token}"
                    attempt += 1
                    continue
                # transient errors → retry with backoff
                if r.status_code in (502, 503, 504, 429):
                    if attempt >= max_retries:
                        r.raise_for_status()
                    sleep = (base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
                    time.sleep(sleep)
                    attempt += 1
                    continue

                r.raise_for_status()
                js = r.json()
                if not js.get("success", False):
                    # treat as empty if API returns success=false
                    return []
                data = js.get("data") or []
                return data if isinstance(data, list) else []
            except requests.RequestException:
                if attempt >= max_retries:
                    raise
                sleep = (base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
                time.sleep(sleep)
                attempt += 1

    out: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_chunk, b) for b in batches]
        for f in as_completed(futs):
            out.extend(f.result())
    return out


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Benchmark YDD queryYunDanDetail throughput")
    ap.add_argument("--csv", default="test/DanHaos.csv", help="Path to CSV containing danHaos")
    ap.add_argument("--column", help="Column name containing danHaos (optional)")
    ap.add_argument("--base", default="http://twc.itdida.com/itdida-api", help="YDD base URL")
    ap.add_argument("--batch-size", type=int, default=10, help="Max 10 per API; obey the limit")
    ap.add_argument("--sleep", type=float, default=0.01, help="Sleep between requests for sequential mode")
    ap.add_argument("--limit", type=int, default=0, help="Only test first N danHaos (0 = all)")
    ap.add_argument("--threads", type=int, default=CNT_THREAD, help="Number of parallel threads (1 = sequential)")
    ap.add_argument("--save", action="store_true", help="Save raw results to output/ydd_results.parquet")
    ap.add_argument("--user", default=YDD_USER, help="YDD username (phone/email)")
    ap.add_argument("--password", default=YDD_PASS, help="YDD password (NOT recommended to pass on CLI)")
    ap.add_argument("--cache", type=bool, default=FLAG_CACHE, help="Flag for applying an existing cache")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    danhaos = load_danhaos(csv_path, column=args.column)
    if args.limit and args.limit > 0:
        danhaos = danhaos[:args.limit]
    if not danhaos:
        print("No danHaos found to benchmark.")
        return
    print(f"Loaded {len(danhaos)} danHaos from {csv_path}")

    # Cache (skip already-known)
    cache_path = Path("output/ydd_ref_map.csv")
    cached: Dict[str, Tuple[str, str]] = {}
    if args.cache and cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cached = {str(r["danHao"]): (str(r["cust_id"]), str(r["transfer_no"])) for _, r in cached_df.iterrows()}
    to_query = [d for d in danhaos if d not in cached]

    # Credentials
    username = args.user or input("YDD username (phone/email): ").strip()
    password = args.password or getpass.getpass("YDD password: ")

    client = YDDClient(base=args.base, username=username, password=password)

    # ---- LOGIN FIRST ----
    t0 = time.perf_counter()
    token = client.login()
    t1 = time.perf_counter()
    print(f"Login OK in {t1 - t0:0.3f}s (token length={len(token)})")

    # ---- QUERY (ONLY to_query) ----
    q0 = time.perf_counter()
    if args.threads > 1:
        items = query_concurrent(
            client,
            to_query,
            batch_size=args.batch_size,
            workers=args.threads,
            max_retries=4,
            base_sleep=0.25,
            jitter=0.15,
        )
    else:
        items = client.query_yundan_detail(to_query, batch_size=args.batch_size, sleep=args.sleep)
    q1 = time.perf_counter()

    elapsed = q1 - q0
    per_danhao = elapsed / max(1, len(to_query)) if to_query else 0.0
    print(f"Queried (not cached): {len(to_query)} with {args.threads} thread(s) in {elapsed:0.3f}s")

    # ---- BUILD MAP + MERGE CACHE ----
    ref2cust = build_ref_to_cust(items)
    ref2cust.update(cached)

    # ---- SAVE/REFRESH CACHE ----
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"danHao": k, "cust_id": v[0], "transfer_no": v[1]} for k, v in ref2cust.items()]
    ).to_csv(cache_path, index=False, encoding="utf-8-sig")

    # ---- MISSING (after merge) + optional one retry ----
    missing_refs = [d for d in danhaos if d not in ref2cust]
    if missing_refs:
        retry_items = client.query_yundan_detail(missing_refs, batch_size=args.batch_size, sleep=0.0)
        ref2cust.update(build_ref_to_cust(retry_items))
        # recompute after retry
        missing_refs = [d for d in danhaos if d not in ref2cust]

    if missing_refs:
        miss_file = Path("output/missing_danhaos.csv")
        miss_file.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(missing_refs, name="danHao").to_csv(miss_file, index=False, encoding="utf-8-sig")
        print(f"❌ Missing danHaos saved to: {miss_file}")
    else:
        print("✅ No missing danHaos.")

    # ---- SUMMARY ----
    print("\n⏱️ Benchmark summary")
    print("-------------------------------------------------")
    print(f"DanHaos total              : {len(danhaos)}")
    print(f"Queried (not cached)       : {len(to_query)}")
    print(f"Threads                    : {args.threads}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Sleep per req (seq mode)   : {args.sleep:0.3f}s")
    print(f"Total query time           : {elapsed:0.3f}s")
    if to_query:
        print(f"Avg per danHao (queried)   : {per_danhao*1000:0.2f} ms/danHao")
        print(f"Throughput (queried)       : {len(to_query)/elapsed:0.2f} danHaos/sec")

    # ---- optional: save raw API items (only the newly queried ones) ----
    if args.save and items:
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(items).to_parquet(out_dir / "ydd_results.parquet", index=False)
        print(f"\nSaved raw items → {out_dir/'ydd_results.parquet'}")


if __name__ == "__main__":
    main()
