# ydd_client.py
from __future__ import annotations
import os, time, logging
from datetime import datetime
from typing import Iterable, Dict, Tuple, List
import requests

YDD_BASE = os.getenv("YDD_BASE", "http://twc.itdida.com/itdida-api")
YDD_USER = os.getenv("YDD_USER", "5055457@qq.com")
YDD_PASS = os.getenv("YDD_PASS", "Twc11434!")
YDD_TIMEOUT = float(os.getenv("YDD_TIMEOUT", "20"))
YDD_ERROR_LOG = os.getenv("YDD_ERROR_LOG", os.path.join("output", "ydd_http_errors.log"))
YDD_MIN_INTERVAL = float(os.getenv("YDD_MIN_INTERVAL", "2"))


class YDDApiHTTPError(RuntimeError):
    """Raised when YDD API returns a non-2xx response with rich diagnostics."""


def _safe_text(response: requests.Response) -> str:
    """Safely get response text for debugging even if decoding fails."""
    try:
        return response.text
    except Exception:
        try:
            return response.content.decode("utf-8", errors="replace")
        except Exception:
            return "<unable to decode response body>"


def _persist_http_error(context: str, response: requests.Response) -> str:
    """Persist full HTTP error details to a local log file for postmortem debugging."""
    ts = datetime.now().isoformat(timespec="seconds")
    body = _safe_text(response)
    request = response.request
    req_headers = dict(request.headers) if request else {}
    res_headers = dict(response.headers)

    # Avoid leaking long bearer tokens into logs while keeping auth scheme info.
    if "Authorization" in req_headers:
        auth_val = str(req_headers["Authorization"])
        req_headers["Authorization"] = auth_val.split(" ")[0] + " <redacted>"

    payload = [
        "=" * 80,
        f"timestamp: {ts}",
        f"context: {context}",
        f"status: {response.status_code}",
        f"reason: {response.reason}",
        f"url: {response.url}",
        f"request_method: {request.method if request else '<unknown>'}",
        f"request_headers: {req_headers}",
        f"response_headers: {res_headers}",
        "response_body_start",
        body,
        "response_body_end",
        "",
    ]

    log_path = os.path.abspath(YDD_ERROR_LOG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(payload))

    return log_path


def _raise_http_error(context: str, response: requests.Response) -> None:
    """Raise a rich exception and persist full response details for server-side failures."""
    log_path = _persist_http_error(context, response)
    body = _safe_text(response)
    raise YDDApiHTTPError(
        f"[{context}] HTTP {response.status_code} {response.reason} for {response.url}\n"
        f"Response body:\n{body}\n"
        f"Full diagnostics appended to: {log_path}"
    )

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
        try:
            r.raise_for_status()
        except requests.HTTPError:
            _raise_http_error("login", r)
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
        try:
            r.raise_for_status()
        except requests.HTTPError:
            _raise_http_error(f"GET {path}", r)
        return r.json()

    # ---- business: query shiment info by order number(10 pkgs/batch) ----
    def query_yundan_detail(self, danhaos: Iterable[str], *, batch_size: int = 10, sleep: float = YDD_MIN_INTERVAL) -> List[dict]:
        """Returns concatenated 'data' arrays across batches."""
        # Ensure we’re authenticated
        if not self.token:
            self.login()

        clean = [str(x).strip() for x in danhaos if str(x).strip()]
        sleep = max(YDD_MIN_INTERVAL, float(sleep))
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
    def query_piece_detail(self, danhaos: Iterable[str], *, batch_size: int = 10, sleep: float = YDD_MIN_INTERVAL) -> List[dict]:
        """Returns concatenated 'data' arrays across batches."""
        # Ensure we’re authenticated
        if not self.token:
            self.login()

        clean = [str(x).strip() for x in danhaos if str(x).strip()]
        sleep = max(YDD_MIN_INTERVAL, float(sleep))
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


