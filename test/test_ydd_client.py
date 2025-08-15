# tests/test_ydd_client.py
from __future__ import annotations
import types
import itertools
import pytest

from YDD_Client import (
    YDDClient,
    build_ref_to_cust,
    build_trk_to_cust,
    select_tracking,
)

# ---------- tiny response helper ----------
class FakeResp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


# ---------- fixtures ----------
@pytest.fixture
def client():
    # Create client with a fake base; we'll replace its session per-test.
    c = YDDClient(base="http://fake/itdida-api", username="u", password="p")
    return c


# ---------- tests ----------
def test_login_sets_token_and_header(client, monkeypatch):
    calls = {}

    def fake_post(url, data=None, timeout=None):
        # verify form fields
        assert url.endswith("/login")
        assert data["username"] == "u"
        assert data["password"] == "p"
        calls["post"] = True
        return FakeResp(200, {"success": True, "data": "TOKEN123"})

    # install fake session
    session = types.SimpleNamespace(
        headers={},
        post=fake_post,
    )
    client.session = session  # inject

    token = client.login()
    assert token == "TOKEN123"
    assert client.token == "TOKEN123"
    assert client.session.headers.get("Authorization") == "Bearer TOKEN123"
    assert calls.get("post") is True


def test_get_auto_relogin_on_401(client, monkeypatch):
    # First GET returns 401; login then second GET returns 200
    state = {"get_calls": 0, "login_calls": 0}

    def fake_post(url, data=None, timeout=None):
        state["login_calls"] += 1
        return FakeResp(200, {"success": True, "data": "NEW_TOKEN"})

    def fake_get(url, params=None, timeout=None):
        state["get_calls"] += 1
        if state["get_calls"] == 1:
            return FakeResp(401, {"success": False})
        return FakeResp(200, {"success": True, "data": []})

    session = types.SimpleNamespace(
        headers={},
        post=fake_post,
        get=fake_get,
    )
    client.session = session

    # no token initially; _get should relogin once
    out = client._get("/queryYunDanDetail", params={"danHaos": "A"})
    assert out == {"success": True, "data": []}
    assert state["get_calls"] == 2  # first 401, then retry 200
    assert state["login_calls"] == 1
    assert client.session.headers["Authorization"] == "Bearer NEW_TOKEN"


def test_query_yundan_detail_batches_and_concatenates(client):
    # Build 23 refs -> should be 3 batches: 10, 10, 3
    refs = [f"REF{i:02d}" for i in range(23)]
    seen_params = []

    # Pretend we're already logged in
    client.token = "X"
    client.session.headers["Authorization"] = "Bearer X"

    def fake_get(url, params=None, timeout=None):
        assert url.endswith("/queryYunDanDetail")
        assert "danHaos" in params
        batch = params["danHaos"].split(",")
        seen_params.append(batch)
        # return a list with one item per ref
        data = [{"keHuDanHao": x, "clientCode": f"C{x[-2:]}", "zhuanDanHao": f"Z{x}"} for x in batch]
        return FakeResp(200, {"success": True, "data": data})

    client.session.get = fake_get

    items = client.query_yundan_detail(refs, batch_size=10, sleep=0.0)

    # verify batch sizes
    sizes = list(map(len, seen_params))
    assert sizes == [10, 10, 3]

    # all refs are returned once
    returned_refs = list(itertools.chain.from_iterable(seen_params))
    assert returned_refs == refs  # same order guaranteed by our fake

    # items length equals 23 (concatenated)
    assert len(items) == 23
    # sample shape
    assert {"keHuDanHao", "clientCode", "zhuanDanHao"} <= set(items[0].keys())


def test_build_ref_to_cust_mapping():
    items = [
        {"keHuDanHao": "REF1", "clientCode": "F000001", "zhuanDanHao": "TRK1"},
        {"keHuDanHao": "REF2", "clientCode": "F000002", "zhuanDanHao": "TRK2"},
        {"keHuDanHao": "", "clientCode": "F000003", "zhuanDanHao": "TRK3"},  # ignored (no ref)
    ]
    m = build_ref_to_cust(items)
    assert m == {
        "REF1": ("F000001", "TRK1"),
        "REF2": ("F000002", "TRK2"),
    }


def test_select_tracking_and_build_trk_to_cust():
    # prefer zhuanDanHao, else upsShipmentId, else seventeenNo, else ""
    i1 = {"clientCode": "F1", "zhuanDanHao": "Z1", "upsShipmentId": "U1", "seventeenNo": "S1"}
    i2 = {"clientCode": "F2", "zhuanDanHao": "",   "upsShipmentId": "U2", "seventeenNo": "S2"}
    i3 = {"clientCode": "F3", "zhuanDanHao": "",   "upsShipmentId": "",   "seventeenNo": "S3"}
    i4 = {"clientCode": "F4", "zhuanDanHao": "",   "upsShipmentId": "",   "seventeenNo": ""}  # no tracking

    assert select_tracking(i1) == "Z1"
    assert select_tracking(i2) == "U2"
    assert select_tracking(i3) == "S3"
    assert select_tracking(i4) == ""

    m = build_trk_to_cust([i1, i2, i3, i4])
    assert m == {
        "Z1": ("F1", "Z1"),
        "U2": ("F2", ""),   # transfer_no is "", function keeps it as provided (or trk if you prefer)
        "S3": ("F3", ""),   # you can change build_trk_to_cust to default transfer_no to chosen tracking
    }
