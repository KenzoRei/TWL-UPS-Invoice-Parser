# ydd_client.py
from __future__ import annotations
import os, time, logging
from typing import Iterable, Dict, Tuple, List
import requests

YDD_BASE = os.getenv("YDD_BASE", "http://twc.itdida.com/itdida-api")
YDD_USER = os.getenv("YDD_USER", "5055457@qq.com")
YDD_PASS = os.getenv("YDD_PASS", "Twc11434!")
YDD_TIMEOUT = float(os.getenv("YDD_TIMEOUT", "20"))

class YDDClient:
    def __init__(self, base: str | None = None, *, username: str | None = None, password: str | None = None):
        self.base = (base or YDD_BASE).rstrip("/")
        self.username = username or YDD_USER
        self.password = password or YDD_PASS
        self.session = requests.Session()
        self.token: str | None = None

    # ---- auth ----
    def login(self) -> str:
        """Login with form-encoded fields and store token (data)."""
        url = f"{self.base}/login"
        payload = {
            "username": self.username,
            "password": self.password,
            # "ssoFlag": "false",  # not required for API
        }
        # form-encoded -> requests uses application/x-www-form-urlencoded for data=
        r = self.session.post(url, data=payload, timeout=YDD_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        if not js.get("success", False):
            raise RuntimeError(f"Login failed: {js}")
        token = (js.get("data") or "").strip()
        if not token:
            raise RuntimeError("Login succeeded but no token in 'data'.")
        self.token = token
        # Set Authorization header for future calls
        self.session.headers["Authorization"] = f"Bearer {token}"
        return token

    # ---- helper: GET with auto 401 relogin ----
    def _get(self, path: str, *, params: dict | None = None, retry: bool = True):
        url = f"{self.base}{path if path.startswith('/') else '/'+path}"
        r = self.session.get(url, params=params or {}, timeout=YDD_TIMEOUT)
        if r.status_code == 401 and retry:
            # token expired → relogin once and retry
            logging.info("YDD 401 received; re-authenticating…")
            self.login()
            return self._get(path, params=params, retry=False)
        r.raise_for_status()
        return r.json()

    # ---- business: query shiment info by order number(10 pkgs/batch) ----
    def query_yundan_detail(self, danhaos: Iterable[str], *, batch_size: int = 10, sleep: float = 0.05) -> List[dict]:
        """Returns concatenated 'data' arrays across batches."""
        # Ensure we’re authenticated
        if not self.token:
            self.login()

        clean = [str(x).strip() for x in danhaos if str(x).strip()]
        out: List[dict] = []
        for i in range(0, len(clean), batch_size):
            chunk = clean[i:i+batch_size]
            # URL encode commas in individual references to prevent API misinterpretation
            encoded_chunk = [ref.replace(",", "%2C") for ref in chunk]
            js = self._get("/queryYunDanDetail", params={"danHaos": ",".join(encoded_chunk)})
            if not js.get("success", False):
                logging.warning(f"[YDD Danhao]YDD success=false for {chunk}: {js}")
            data = js.get("data") or []
            if not isinstance(data, list):
                logging.warning("[YDD Danhao]YDD payload 'data' is not a list; skipping.")
                data = []
            out.extend(data)
            time.sleep(sleep)
        return out

    # ---- business: query package info by tracking number(10 pkgs/batch) ----
    # Inputs can be Reference Number or Lead Shipment Number
    def query_piece_detail(self, danhaos: Iterable[str], *, batch_size: int = 10, sleep: float = 0.05) -> List[dict]:
        """Returns concatenated 'data' arrays across batches."""
        # Ensure we’re authenticated
        if not self.token:
            self.login()

        clean = [str(x).strip() for x in danhaos if str(x).strip()]
        out: List[dict] = []
        for i in range(0, len(clean), batch_size):
            chunk = clean[i:i+batch_size]
            # URL encode commas in individual references to prevent API misinterpretation
            encoded_chunk = [ref.replace(",", "%2C") for ref in chunk]
            js = self._get("/queryPieceDetail", params={"danHaos": ",".join(encoded_chunk)})
            if not js.get("success", False):
                logging.warning(f"[YDD Trk]YDD success=false for {chunk}: {js}")
            data = js.get("data") or []
            if not isinstance(data, list):
                logging.warning("[YDD Trk]YDD payload 'data' is not a list; skipping.")
                data = []
            out.extend(data)
            time.sleep(sleep)
        return out

# ------- convenience mappers for your parser -------

def select_tracking(d: dict) -> str:
    """Prefer 转单号→UPS shipment id→17单号→queryBillNo→carrierNo as the tracking/lead id."""
    for k in ("zhuanDanHao", "queryBillNo", "carrierNo", "upsShipmentId", "seventeenNo"):
        v = str(d.get(k, "")).strip()
        if v:
            return v
    return ""

def build_ref_to_cust(items: List[dict]) -> Dict[str, Tuple[str, str]]:
    """
    Map by 客户单号(keHuDanHao) → (cust_id(clientCode), transfer_no(zhuanDanHao)).
    Useful to join on 'Shipment Reference Number 1' from UPS CSVs.
    """
    m: Dict[str, Tuple[str, str]] = {}
    for it in items:
        ref = str(it.get("keHuDanHao", "")).strip()
        cust = str(it.get("clientCode", "")).strip()
        transfer = str(it.get("zhuanDanHao", "")).strip()
        if ref and cust:
            m[ref] = (cust, transfer)
    return m

def build_trk_to_cust(items: List[dict]) -> Dict[str, Tuple[str, str]]:
    """
    Map by chosen tracking (转单号优先) → (cust_id, transfer_no).
    Useful to join on Tracking Number / Lead Shipment Number.
    """
    m: Dict[str, Tuple[str, str]] = {}
    for it in items:
        trk = select_tracking(it)
        cust = str(it.get("clientCode", "")).strip()
        transfer = str(it.get("zhuanDanHao", "")).strip()
        if trk and cust:
            m[trk] = (cust, transfer or trk)
    return m
