"""UPS Customer Matcher for matching invoices to customers."""

import datetime
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from config import (
    SPECIAL_ACCT_TO_CUST,
    SPECIAL_CUSTOMERS,
    General_Cost_EN,
    FLAG_API_USE,
    FLAG_DEBUG,
    YDD_USER,
    YDD_PASS,
)
from utils.helpers import is_blank


class UpsCustomerMatcher:
    """Match UPS invoice charges to customers using YDD API or manual mapping."""
    
    def __init__(
        self,
        normalized_df: pd.DataFrame,
        mapping_file: Optional[Path] = None,
        *,
        use_api: bool = FLAG_API_USE,
        ydd_threads: int = 1,
        ydd_batch_size: int = 9,
        ydd_client: Optional[object] = None,
        use_cache: bool = True,
    ):
        """
        Initialize customer matcher.
        
        Args:
            normalized_df: DataFrame output from normalizer.get_normalized_data()
            mapping_file: 数据列表*.xlsx (only used when use_api=False)
            use_api: True -> fetch mapping via YDD API; False -> use manual Excel mapping
            ydd_threads: Parallel threads for YDD API (1 = sequential)
            ydd_batch_size: Max refs per request (YDD docs say 10)
            ydd_client: Optional preconfigured YDDClient; if None, env vars are used
            use_cache: If True, cache danHao→(cust_id,tracking) to output/ydd_ref_map.csv
        """
        self.df = normalized_df.copy()
        if "cust_id" not in self.df.columns:
            self.df["cust_id"] = ""
        if "Charge_Cate_EN" not in self.df.columns:
            self.df["Charge_Cate_EN"] = ""
        if "Charge_Cate_CN" not in self.df.columns:
            self.df["Charge_Cate_CN"] = ""

        self.DEFAULT_CUST_ID = "F000999"
        self.base_path = Path(__file__).resolve().parent.parent

        # Mode / knobs
        self.use_api = use_api
        self.ydd_threads = max(1, int(ydd_threads))
        self.ydd_batch_size = max(1, int(ydd_batch_size))
        self._ydd_client = ydd_client
        self.use_cache = use_cache

        # Manual mapping fields
        self.mapping_file: Optional[Path] = mapping_file
        self.mapping_cust_df = pd.DataFrame()
        self.trk_to_cust: Dict[str, Tuple[str, str]] = {}  # Tracking -> (cust_id, LeadShipment)

        # API mapping fields
        self.ref_to_cust: Dict[str, Tuple[str, str]] = {}  # danHao -> (cust_id, chosen_tracking)

        # Shared mappings (csv)
        self.mapping_pickup = self.base_path / "data" / "mappings" / "Pickups.csv"
        self.mapping_chrg = self.base_path / "data" / "mappings" / "Charges.csv"
        self.mapping_ar = self.base_path / "data" / "mappings" / "ARCalculator.csv"

        self.mapping_pickup_df = pd.DataFrame()
        self.mapping_chrg_df = pd.DataFrame()
        self.mapping_ar_df = pd.DataFrame()

        self.dict_pickup: Dict[str, dict] = {}
        self.dict_chrg: Dict[str, dict] = {}
        self.dict_ar: Dict[str, dict] = {}

        # YDD cache file
        self.api_cache_path = self.base_path / "output" / "ydd_ref_map.csv"

        # Exception export
        self.excluded_from_exception = set(SPECIAL_ACCT_TO_CUST.keys())

        self.api_stats: dict = {}

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _best_tracking_for_row(self, row: pd.Series) -> str:
        """Get best tracking number: prefer Lead Shipment, fallback to Tracking."""
        ls = str(row.get("Lead Shipment Number", "") or "").strip()
        trk = str(row.get("Tracking Number", "") or "").strip()
        return ls if not is_blank(ls) else trk

    def set_mapping_file(self, path: Path) -> None:
        """Set the manual mapping file path."""
        self.mapping_file = Path(path) if path else None

    def choose_mapping_file_dialog(self) -> Optional[Path]:
        """Open dialog to select manual mapping Excel file."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            raise RuntimeError("tkinter is not available for file selection.") from e
        
        root = tk.Tk()
        root.withdraw()
        fp = filedialog.askopenfilename(
            title="选择数据列表*.xlsx 映射文件 / Select mapping file",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        root.update()
        root.destroy()
        self.mapping_file = Path(fp) if fp else None
        return self.mapping_file

    # ============================================================================
    # MAPPING LOADING
    # ============================================================================

    def _load_common_mappings(self) -> None:
        """Load common CSV mappings (charges, pickups, AR calculator)."""
        self.mapping_chrg_df = pd.read_csv(self.mapping_chrg)
        self.dict_chrg = self.mapping_chrg_df.set_index(
            self.mapping_chrg_df.columns[0]
        ).to_dict(orient="index")

        self.mapping_pickup_df = pd.read_csv(self.mapping_pickup)
        self.dict_pickup = self.mapping_pickup_df.set_index(
            self.mapping_pickup_df.columns[0]
        ).to_dict(orient="index")

        self.mapping_ar_df = pd.read_csv(self.mapping_ar)
        self.dict_ar = self.mapping_ar_df.set_index(
            self.mapping_ar_df.columns[0]
        ).to_dict(orient="index")

    def _load_mapping_manual(self) -> None:
        """Load manual Excel mapping file."""
        if self.mapping_file is None:
            raise FileNotFoundError(
                "Mapping file not set. Call choose_mapping_file_dialog() or set_mapping_file()."
            )
        
        self.mapping_cust_df = pd.read_excel(self.mapping_file)
        required_cols = {"子转单号", "客户编号", "转单号"}
        if not required_cols.issubset(self.mapping_cust_df.columns):
            missing = required_cols - set(self.mapping_cust_df.columns)
            raise ValueError(f"Mapping file missing required columns: {missing}")

        df = self.mapping_cust_df.copy()
        df["子转单号"] = (
            df["子转单号"].astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )
        df = df.assign(Tracking=df["子转单号"].str.split(",")).explode("Tracking")
        df["Tracking"] = df["Tracking"].astype(str).str.replace(" ", "", regex=False)
        df = df[df["Tracking"] != ""]

        # Tracking -> (cust, LeadShipment/转单号)
        self.trk_to_cust = dict(
            zip(df["Tracking"], zip(df["客户编号"].astype(str), df["转单号"].astype(str)))
        )
        self._load_common_mappings()

    def _ensure_ydd_client(self):
        """Ensure YDD client is initialized."""
        if self._ydd_client is not None:
            return self._ydd_client
        
        base = os.environ.get("YDD_BASE", "http://twc.itdida.com/itdida-api")
        user = os.environ.get("YDD_USER", YDD_USER)
        pwd = os.environ.get("YDD_PASS", YDD_PASS)
        
        if not user or not pwd:
            raise RuntimeError(
                "YDD creds missing. Set env YDD_USER and YDD_PASS (opt YDD_BASE)."
            )
        
        from YDD_Client import YDDClient
        self._ydd_client = YDDClient(base=base, username=user, password=pwd)
        return self._ydd_client

    def _collect_danhaos_with_tracking(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Build list of danhaos and tracking mapping.
        
        Returns:
            Tuple of (danhaos list, ref_to_best_trk dict)
            - danhaos: unique Shipment Reference Number 1 values
            - ref_to_best_trk: danHao -> chosen_tracking (LeadShipment preferred, else Tracking)
            
        Only includes rows where Account Number is NOT a special account.
        """
        df = self.df.copy()
        specials = set(SPECIAL_ACCT_TO_CUST.keys())

        if FLAG_DEBUG:
            print(f"[DEBUG] Specials (excluded from YDD): {specials}")
        
        mask = ~df["Account Number"].astype(str).isin(specials)
        excluded_accounts = df.loc[~mask, "Account Number"].unique().tolist()
        if FLAG_DEBUG:
            print(f"[DEBUG] Excluded Account Numbers: {excluded_accounts}")
        
        included_accounts = df.loc[mask, "Account Number"].unique().tolist()
        if FLAG_DEBUG:
            print(f"[DEBUG] Included Account Numbers (sent to YDD): {included_accounts}")

        sub = df.loc[
            mask, ["Shipment Reference Number 1", "Lead Shipment Number", "Tracking Number"]
        ].copy()
        sub["Shipment Reference Number 1"] = sub["Shipment Reference Number 1"].astype(str).str.strip()
        sub = sub[sub["Shipment Reference Number 1"].apply(lambda x: not is_blank(x))]

        sub["best_trk"] = sub.apply(self._best_tracking_for_row, axis=1)
        sub = sub.drop_duplicates(subset=["Shipment Reference Number 1"], keep="first")

        refs = sub["Shipment Reference Number 1"].tolist()
        if FLAG_DEBUG:
            df.loc[mask].to_excel(
                self.base_path / "output" / "ydd_refs_sent.xlsx", index=False
            )
            print(f"[DEBUG] Shipment sent to YDD saved to output/ydd_refs_sent.xlsx")
        
        ref_to_best_trk = dict(zip(sub["Shipment Reference Number 1"], sub["best_trk"]))
        return refs, ref_to_best_trk

    @staticmethod
    def _query_concurrent(
        client,
        danhaos: List[str],
        *,
        batch_size: int,
        workers: int,
        max_retries: int = 6,
        base_sleep: float = 0.5,
        jitter: float = 0.3,
    ) -> List[dict]:
        """
        Parallel /queryYunDanDetail with per-thread Session and retry/backoff.
        """
        batches = [danhaos[i : i + batch_size] for i in range(0, len(danhaos), batch_size)]
        if not batches:
            return []
        if not client.token:
            raise RuntimeError("Call client.login() before _query_concurrent().")

        auth_header = {"Authorization": f"Bearer {client.token}"}
        base_url = client.base.rstrip("/") + "/queryYunDanDetail"
        default_timeout = 20

        def fetch_chunk(chunk: List[str]) -> List[dict]:
            params = {"danHaos": ",".join(chunk)}
            s = requests.Session()
            s.headers.update(auth_header)
            attempt = 0
            while True:
                try:
                    r = s.get(base_url, params=params, timeout=default_timeout)
                    if r.status_code == 401:
                        client.login()
                        s.headers["Authorization"] = f"Bearer {client.token}"
                        attempt += 1
                        continue
                    if r.status_code in (502, 503, 504, 429):
                        if attempt >= max_retries:
                            r.raise_for_status()
                        sleep = (base_sleep * (2**attempt)) + random.uniform(0, jitter)
                        time.sleep(sleep)
                        attempt += 1
                        continue
                    r.raise_for_status()
                    js = r.json()
                    if not js.get("success", False):
                        return []
                    data = js.get("data") or []
                    return data if isinstance(data, list) else []
                except requests.RequestException:
                    if attempt >= max_retries:
                        raise
                    sleep = (base_sleep * (2**attempt)) + random.uniform(0, jitter)
                    time.sleep(sleep)
                    attempt += 1

        out: List[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch_chunk, b) for b in batches]
            for f in as_completed(futs):
                out.extend(f.result())
        return out

    def _load_api_cache(self) -> Dict[str, Tuple[str, str]]:
        """Load API cache from disk if available."""
        if not self.use_cache:
            return {}
        
        p = self.api_cache_path
        try:
            if not p.exists() or p.stat().st_size == 0:
                return {}
        except Exception:
            return {}

        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            logging.warning(f"[YDD] Cache file is empty: {p}; ignoring.")
            return {}
        except Exception as e:
            logging.warning(f"[YDD] Failed to read cache {p}: {e}; ignoring.")
            return {}

        required = {"danHao", "cust_id", "tracking"}
        if not required.issubset(df.columns):
            logging.warning(
                f"[YDD] Cache missing columns {required - set(df.columns)}; ignoring."
            )
            return {}

        return {
            str(r["danHao"]): (str(r["cust_id"]), str(r["tracking"]))
            for _, r in df.iterrows()
        }

    def _save_api_cache(self, ref_to_cust: Dict[str, Tuple[str, str]]) -> None:
        """Save API mapping to cache file."""
        if not self.use_cache:
            return
        
        self.api_cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"danHao": k, "cust_id": v[0], "tracking": v[1]} for k, v in ref_to_cust.items()
        ]
        df = pd.DataFrame(rows)
        tmp = self.api_cache_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(self.api_cache_path)

    def _load_mapping_api(self) -> None:
        """Load customer mapping via YDD API."""
        from YDD_Client import build_ref_to_cust
        
        client = self._ensure_ydd_client()

        # Login timing
        t0 = time.perf_counter()
        token = client.login()
        t1 = time.perf_counter()

        danhaos, ref_to_best_trk = self._collect_danhaos_with_tracking()
        total_refs = len(danhaos)

        # Load cache
        cached = self._load_api_cache()
        cached_hits = sum(1 for d in danhaos if d in cached)
        to_query = [d for d in danhaos if d not in cached]

        print(f"[YDD] Login OK in {t1 - t0:0.3f}s (token len={len(token)})")
        print(
            f"[YDD] Refs total={total_refs}, cached={cached_hits}, querying={len(to_query)}, "
            f"threads={self.ydd_threads}, batch_size={min(self.ydd_batch_size, 10)}"
        )

        api_items = []
        error_msg = None
        try:
            if to_query:
                if self.ydd_threads > 1:
                    api_items = self._query_concurrent(
                        client,
                        to_query,
                        batch_size=min(self.ydd_batch_size, 10),
                        workers=self.ydd_threads,
                    )
                else:
                    api_items = client.query_yundan_detail(
                        to_query, batch_size=min(self.ydd_batch_size, 10), sleep=0.01
                    )
            
            ref2api = build_ref_to_cust(api_items)  # danHao -> (cust_id, transfer_no)

            if FLAG_DEBUG:
                pd.DataFrame(
                    [
                        {"danhao": k, "cust_id": v[0], "transfer_no": v[1]}
                        for k, v in ref2api.items()
                    ]
                ).to_excel(self.base_path / "output" / "ref2api_check.xlsx", index=False)
                print(f"[DEBUG] YDD API results saved to output/ref2api_check.xlsx")

            # Normalize to (cust_id, transfer_no) -- always use API's transfer_no
            fresh_ref_to_cust = {
                ref: (cid, xfer) for ref, (cid, xfer) in ref2api.items()
            }
            cached.update(fresh_ref_to_cust)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[YDD] ❌ API error: {error_msg}")

        # Finalize
        self.ref_to_cust = cached
        self._save_api_cache(self.ref_to_cust)
        self._load_common_mappings()

        # Compute missing & write CSV if any
        missing = [d for d in danhaos if d not in self.ref_to_cust]
        if missing:
            miss_path = self.base_path / "output" / "missing_danhaos_ydd.csv"
            miss_path.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(missing, name="danHao").to_csv(
                miss_path, index=False, encoding="utf-8-sig"
            )
            print(f"[YDD] ❌ Missing {len(missing)} ref(s). Saved to: {miss_path}")
        else:
            print("[YDD] ✅ All refs matched by API/cache.")

        # Store stats for caller/UI/tests
        self.api_stats = {
            "total_refs": total_refs,
            "cached_hits": cached_hits,
            "queried": len(to_query),
            "api_items": len(api_items),
            "mapped_final": len(self.ref_to_cust),
            "missing_count": len(missing),
            "missing_sample": missing[:10],
            "error": error_msg,
        }

    def _load_mapping(self) -> None:
        """Load mapping (API or manual)."""
        if self.use_api:
            self._load_mapping_api()
        else:
            self._load_mapping_manual()
        
        # For debugging: dump ref_to_cust mapping
        if FLAG_DEBUG and self.use_api:
            pd.DataFrame(
                [
                    {"danhao": k, "cust_id": v[0], "lead_shipment": v[1]}
                    for k, v in self.ref_to_cust.items()
                ]
            ).to_excel("output/ref_to_cust_check.xlsx", index=False)
            print(f"[DEBUG] ref_to_cust mapping saved to output/ref_to_cust_check.xlsx")

    # ============================================================================
    # MAIN WORKFLOW
    # ============================================================================

    def match_customers(self) -> None:
        """
        Main customer matching workflow.
        
        Steps:
        1. Load mappings (API or manual)
        2. Pre-assign special customers by account
        3. Classify charges and match customers
        4. Calculate AR amounts
        5. Generate exception template
        6. Export unmapped charges
        """
        # Ensure mappings are ready
        if not (self.dict_chrg and self.dict_pickup and self.dict_ar) and not self.ref_to_cust and not self.trk_to_cust:
            self._load_mapping()

        # 0) Pre-assign specials by account
        self.df["cust_id"] = self.df["cust_id"].astype(str)
        self.df["cust_id_special"] = self.df["Account Number"].map(SPECIAL_ACCT_TO_CUST)
        mask_special = self.df["cust_id_special"].notna()
        self.df.loc[mask_special, "cust_id"] = self.df.loc[mask_special, "cust_id_special"]
        self.df.drop(columns=["cust_id_special"], inplace=True)

        exception_rows = []
        for idx, row in self.df.iterrows():
            cust_id, lead_shipment = np.nan, np.nan
            exception_flag = False

            # Classify charges
            category_en, category_cn = self._charge_classifier(row)
            self.df.at[idx, "Charge_Cate_EN"] = category_en
            row["Charge_Cate_EN"] = category_en
            self.df.at[idx, "Charge_Cate_CN"] = category_cn
            row["Charge_Cate_CN"] = category_cn

            # HARD RULE: general costs never go to customers
            if category_en in General_Cost_EN:
                self.df.at[idx, "cust_id"] = self.DEFAULT_CUST_ID
                row["cust_id"] = self.DEFAULT_CUST_ID
                continue

            if self.use_api:
                danhao = str(row.get("Shipment Reference Number 1", "") or "").strip()
                if (
                    not is_blank(danhao)
                    and danhao in self.ref_to_cust
                    and row["Account Number"] not in self.excluded_from_exception
                ):
                    cust_id, chosen_trk = self.ref_to_cust[danhao]
                    lead_shipment = (
                        chosen_trk
                        if not is_blank(chosen_trk)
                        else self._best_tracking_for_row(row)
                    )
                else:
                    cust_id = self._exception_handler(row)
                    exception_flag = True
                    lead_shipment = self._best_tracking_for_row(row)
            else:
                trk_num = str(row.get("Tracking Number", "") or "")
                if trk_num in self.trk_to_cust:
                    cust_id, lead_shipment = self.trk_to_cust[trk_num]
                else:
                    cust_id = self._exception_handler(row)
                    exception_flag = True
                    lead_shipment = self._best_tracking_for_row(row)

            # Write back
            self.df.at[idx, "cust_id"] = cust_id
            self.df.at[idx, "Lead Shipment Number"] = lead_shipment
            if is_blank(row.get("Tracking Number", "")):
                self.df.at[idx, "Tracking Number"] = lead_shipment

            # Backfill when needed
            if is_blank(self.df.at[idx, "Lead Shipment Number"]) and cust_id != self.DEFAULT_CUST_ID:
                ls = str(row.get("Invoice Number", "")) + "-AcctExpense"
                self.df.at[idx, "Lead Shipment Number"] = ls
                self.df.at[idx, "Tracking Number"] = ls
                self.df.at[idx, "Shipment Reference Number 1"] = ls

            if exception_flag and not is_blank(self.df.at[idx, "Lead Shipment Number"]):
                exception_rows.append(self.df.loc[idx].to_dict())

        # AR calc + templates + unmapped logging
        self._ar_calculator()

        df_exception = pd.DataFrame(exception_rows)
        if not df_exception.empty:
            num_cols = ["总实重", "长", "宽", "高"]
            for col in df_exception.columns:
                if col in num_cols:
                    df_exception[col] = pd.to_numeric(df_exception[col], errors="coerce").fillna(0)
                else:
                    df_exception[col] = df_exception[col].replace({np.nan: ""}).replace({"nan": ""})
            self._template_generator(df_exception)

        unmapped = self.df[self.df["Charge_Cate_EN"] == "Not Defined Charge"]
        if not unmapped.empty:
            out = self.base_path / "output" / "UnmappedCharges.xlsx"
            out.parent.mkdir(parents=True, exist_ok=True)
            unmapped.to_excel(out, index=False)

        # Optional: api stats summary
        if self.use_api:
            s = self.api_stats or {}
            print("\n[YDD] API summary")
            print("-" * 50)
            print(f"Refs total     : {s.get('total_refs', 0)}")
            print(f"Cached hits    : {s.get('cached_hits', 0)}")
            print(f"Queried        : {s.get('queried', 0)}")
            print(f"API items      : {s.get('api_items', 0)}")
            print(f"Mapped (final) : {s.get('mapped_final', 0)}")
            print(f"Missing        : {s.get('missing_count', 0)}")
            if s.get("error"):
                print(f"Last error     : {s['error']}")

    def get_matched_data(self) -> pd.DataFrame:
        """Return the updated DataFrame with customer info matched."""
        return self.df

    # ============================================================================
    # EXCEPTION HANDLING
    # ============================================================================

    def _exception_handler(self, row: pd.Series) -> str:
        """
        Manually match a shipment with cust_id and return it.
        
        Args:
            row: DataFrame row to match
            
        Returns:
            Customer ID string
        """
        # 0. General cost
        if row["Charge_Cate_EN"] in General_Cost_EN:
            return self.DEFAULT_CUST_ID

        # 1. Pass cust_id if it already exists
        if not is_blank(row["cust_id"]) and row["cust_id"] != self.DEFAULT_CUST_ID:
            return row["cust_id"]

        # 2. Import department pickup
        elif (
            row["Lead Shipment Number"].startswith("2")
            and "import" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000223"

        # 3. UPEX (vermilion match)
        elif (
            "vermilion" in row["Sender Name"].lower()
            or "vermilion" in row["Sender Company Name"].lower()
            or "vermilion" in row["Shipment Reference Number 1"].lower()
            or "yuzhao liu" in row["Sender Name"].lower()
            or "yuzhao liu" in row["Sender Company Name"].lower()
            or "xiaorong" in row["Shipment Reference Number 1"].lower()
        ):
            return "F000215"

        # 4. Bondex (pickup fee allocates to TWW)
        elif (
            row["Account Number"] in ["K5811C", "F03A44"]
            and row["Charge_Cate_EN"] not in ["Daily Pickup", "Daily Pickup - Fuel"]
        ):
            return "F000281"

        # 5. SPT
        elif row["Account Number"] == "HE6132":
            return "F000208"

        # 6. Transware
        elif row["Account Number"] in (["H930G2", "H930G3", "H930G4", "R1H015", "XJ3936"]):
            return "F000222"
        elif not is_blank(row["Tracking Number"]) or not is_blank(
            row["Lead Shipment Number"]
        ):
            return "F000222"

        # 7. Daily Pickup logic
        elif row["Charge_Cate_EN"] in ["Daily Pickup", "Daily Pickup - Fuel"]:
            return self.dict_pickup.get(row["Account Number"], {}).get(
                "Cust.ID", self.DEFAULT_CUST_ID
            )

        # 8. Generic cost rules
        elif str(row["Charge_Cate_EN"]).upper() in ["SCC AUDIT FEE", "POD FEE"]:
            return self.DEFAULT_CUST_ID

        # 9. Fallback
        print(f"account number: {row['Account Number']}")
        print(f"Charge_Cate_EN: {row['Charge_Cate_EN']}")
        print(
            f"pass all exception handlings. 'Charge Description':{row['Charge Description']}, "
            f"'Charge_Cate_EN':{row['Charge_Cate_EN']}, 'Amount':{row['Net Amount']}"
        )
        return self.DEFAULT_CUST_ID

    # ============================================================================
    # CHARGE CLASSIFICATION
    # ============================================================================

    def _charge_classifier(self, row: pd.Series) -> Tuple[str, str]:
        """
        Classify charge by "charge description" and all types of class codes.
        
        Args:
            row: DataFrame row to classify
            
        Returns:
            Tuple of (category_en, category_cn)
        """
        ChrgDesc = row["Charge Description"]
        Chrg_EN = "Not Defined Charge"
        Chrg_CN = "未分类费用"
        
        for term in [
            "Not Previously Billed ",
            "Void ",
            "Shipping Charge Correction ",
            " Adjustment",
            "ZONE ADJUSTMENT ",
            "Returns ",
        ]:
            ChrgDesc = ChrgDesc.replace(term, "")

        # IMPORTANT: CHARGE CLASSIFICATION LOGIC HERE
        if is_blank(row["Tracking Number"]) and row["Charge Category Detail Code"] == "SVCH":
            if row["Charge Description"] == "Payment Processing Fee":
                Chrg_EN = "Payment Processing Fee"
                Chrg_CN = "信用卡手续费"
            elif row["Charge Description"] == "Fuel Surcharge":
                Chrg_EN = "Daily Pickup - Fuel"
                Chrg_CN = "每日取件-燃油"
            elif row["Charge Description"] == "Service Charge":
                Chrg_EN = "Daily Pickup"
                Chrg_CN = "每日取件"
        elif ChrgDesc in self.dict_chrg:
            Chrg_EN = self.dict_chrg[ChrgDesc]["Charge_Cate_EN"]
            Chrg_CN = self.dict_chrg[ChrgDesc]["Charge_Cate_CN"]
        elif (
            row["Charge Category Detail Code"] == "CADJ"
            and row["Charge Classification Code"] == "MSC"
        ):
            if "audit fee" in row["Miscellaneous Line 1"].lower():
                Chrg_EN = "SCC Audit Fee"
                Chrg_CN = "UPS SCC审计费"
            else:
                Chrg_EN = "Shipping Charge Correction"
                Chrg_CN = "费用修正"
        elif row["Charge Category Detail Code"] == "FPOD":
            Chrg_EN = "POD Fee"
            Chrg_CN = "送抵证明"

        return Chrg_EN, Chrg_CN

    # ============================================================================
    # TEMPLATE GENERATION
    # ============================================================================

    def _template_generator(self, df: pd.DataFrame):
        """Generate an YiDiDa template and save to output folder."""
        # Filter for valid exception rows
        cols_to_check = ["Lead Shipment Number", "Tracking Number"]
        condition_missing = df[cols_to_check].apply(
            lambda row: any(is_blank(val) for val in row), axis=1
        )
        condition_exclude = ~df["Account Number"].isin(self.excluded_from_exception)
        df_exceptions = df[~condition_missing & condition_exclude]
        df_exceptions = df_exceptions.drop_duplicates(
            subset=["Lead Shipment Number", "Tracking Number"], keep="first"
        )
        
        col_names = [
            "客户代码", "转单号", "承运商子单号", "客户单号(文本格式)",
            "收货渠道", "目的地国家", "件数", "总实重", "长", "宽",
            "高", "省份/洲名简码", "城市", "邮编", "收件公司",
            "收件人姓名", "收件人电话", "收件人地址一", "收件人地址二",
            "包裹类型", "报关方式", "付税金", "中文品名", "英文品名",
            "海关编码", "数量", "币种", "单价", "总价", "购买保险",
            "寄件人姓名", "寄件公司", "寄件电话", "寄件人地址一", "寄件人地址二",
            "寄件人城市", "寄件人州名", "寄件人国家", "寄件人邮编",
            "货物特性", "收货备注", "预报备注",
        ]
        
        output_template = pd.DataFrame(columns=col_names)
        output_template["客户代码"] = df_exceptions["cust_id"].fillna("F000999").astype(str)
        output_template["转单号"] = df_exceptions["Lead Shipment Number"].fillna("UNKNOWN").astype(str)
        output_template["承运商子单号"] = df_exceptions["Tracking Number"].fillna("UNKNOWN").astype(str)
        output_template["客户单号(文本格式)"] = (
            df_exceptions["Shipment Reference Number 1"].fillna("NoRef").astype(str)
            + "-"
            + df_exceptions["Lead Shipment Number"].fillna("UNKNOWN").astype(str)
        ).str[:34]
        output_template["收货渠道"] = "UPS Exception Handling"
        output_template["目的地国家"] = "US"
        output_template["件数"] = 1
        output_template["总实重"] = (
            pd.to_numeric(df_exceptions["Billed Weight"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x / 2.204, 1))
        ).round(2)
        output_template["长"] = (
            pd.to_numeric(df_exceptions["Billed Length"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["宽"] = (
            pd.to_numeric(df_exceptions["Billed Width"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["高"] = (
            pd.to_numeric(df_exceptions["Billed Height"], errors="coerce")
            .fillna(0)
            .apply(lambda x: max(x * 2.56, 1))
        ).round(2)
        output_template["省份/洲名简码"] = df_exceptions["Receiver State"].replace("", "CA").astype(str)
        output_template["城市"] = df_exceptions["Receiver City"].replace("", "UNKNOWN").astype(str)
        output_template["邮编"] = df_exceptions["Receiver Postal"].replace("", "90248").astype(str)
        output_template["收件公司"] = df_exceptions["Receiver Company Name"].replace("", "UNKNOWN").astype(str)
        output_template["收件人姓名"] = df_exceptions["Receiver Name"].replace("", "UNKNOWN").astype(str)
        output_template["收件人地址一"] = df_exceptions["Receiver Address Line 1"].replace("", "UNKNOWN").astype(str)
        output_template["收件人地址二"] = df_exceptions["Receiver Address Line 2"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人姓名"] = df_exceptions["Sender Name"].replace("", "UNKNOWN").astype(str)
        output_template["寄件公司"] = df_exceptions["Sender Company Name"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人地址一"] = df_exceptions["Sender Address Line 1"].fillna("UNKNOWN").astype(str)
        output_template["寄件人地址二"] = df_exceptions["Sender Address Line 2"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人城市"] = df_exceptions["Sender City"].replace("", "UNKNOWN").astype(str)
        output_template["寄件人州名"] = df_exceptions["Sender State"].replace("", "CA").astype(str)
        output_template["寄件人国家"] = "US"
        output_template["寄件人邮编"] = df_exceptions["Sender Postal"].replace("", "90248").astype(str)
        output_template["预报备注"] = "ManualImport " + datetime.date.today().strftime("%Y-%m-%d")
        
        for col in [
            "报关方式", "付税金", "中文品名", "英文品名", "海关编码",
            "数量", "币种", "单价", "总价", "购买保险", "货物特性", "收货备注",
        ]:
            output_template[col] = ""
        
        if not output_template.empty:
            output_path = self.base_path / "output" / "ExceptionImport_YDD.xlsx"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_template = output_template.fillna("").replace("nan", "")
            output_template.to_excel(output_path, index=False)
            print(f"✅ YDD Exception Template generated with {len(output_template)} rows.")
            print(f"📁 YDD Exception Template Saved to: {output_path}")

    # ============================================================================
    # AR CALCULATION
    # ============================================================================

    def _ar_calculator(self):
        """Calculate AR amount according to cust_id."""
        # Verify self.df
        empty_cust_id_rows = self.df[self.df["cust_id"].apply(is_blank)]
        if not empty_cust_id_rows.empty:
            print(f"[Warning] {len(empty_cust_id_rows)} rows have empty cust_id.")
        
        invalid_cust_id_rows = self.df[~self.df["cust_id"].isin(self.dict_ar.keys())]
        if not invalid_cust_id_rows.empty:
            print(
                f"[Warning] {len(invalid_cust_id_rows)} rows have unmapped cust_id "
                f"(not in AR mapping)."
            )

        # Calculate AR Amount using business rules and AR factor mapping
        self.df["AR_Factor"] = self.df["cust_id"].map(
            lambda cid: self.dict_ar.get(cid, {}).get("Factor", 0.0)
        )
        self.df["Flag_Modifier"] = self.df["cust_id"].map(
            lambda cid: self.dict_ar.get(cid, {}).get("Flag_Modifier", "")
        )

        # Business rule — if SIM + negative + no modifier → AR = 0
        cond_special_zero = (
            (self.df["Charge_Cate_EN"] == "Special Incentive Modifier")
            & (self.df["Net Amount"] < 0)
            & (self.df["Flag_Modifier"].apply(is_blank))
        )

        # Default AR amount = Net × Factor
        self.df["AR_Amount"] = (self.df["Net Amount"] * self.df["AR_Factor"]).round(2)

        # Apply override where condition is met
        self.df.loc[cond_special_zero, "AR_Amount"] = 0.00
