import json

import pandas as pd
import pytest
import requests

import config
import data_fetcher

# Shaped like the documented city-listings response.
PAYLOAD = {
    "success": True,
    "meta": {"billable": True, "endpoint_family": "city_listings"},
    "data": {
        "city": "Bangalore",
        "properties": [
            {"id": "A1", "url": "https://www.99acres.com/a1", "city": "Bangalore East",
             "locality": "Whitefield, Bangalore", "area": {"unit": "sq.ft", "value": 1200},
             "price": {"min": 7500000, "max": 7500000, "currency": "INR"}, "bedrooms": 2, "bathrooms": 2,
             "transaction_type": "sale", "property_category": "residential"},
            {"id": "A2", "url": "https://www.99acres.com/a2", "city": "Bangalore South",
             "locality": "Jp Nagar, Bangalore", "area": {"unit": "sq.yards", "value": 200},
             "price": {"min": 15000000, "max": 17000000, "currency": "INR"}, "bedrooms": 3, "bathrooms": 3},
            {"id": "BAD-PRICE", "area": {"unit": "sq.ft", "value": 1000}, "price": {"min": 0, "max": 0},
             "bedrooms": 2, "locality": "X, Bangalore"},
            {"id": "BAD-UNIT", "area": {"unit": "hectare", "value": 1}, "price": {"min": 5000000},
             "bedrooms": 2, "locality": "X, Bangalore"},
            {"id": "RENT", "transaction_type": "rent", "area": {"unit": "sq.ft", "value": 900},
             "price": {"min": 25000}, "bedrooms": 2, "locality": "X, Bangalore"},
        ],
    },
}


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body
        self.headers = {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setattr(config, "SCRAPPA_API_KEY", "test-key")
    monkeypatch.setattr(config, "MAX_RETRIES", 2)
    monkeypatch.setattr(data_fetcher.time, "sleep", lambda s: None)


def test_normalise_parses_and_rejects():
    df, rejected = data_fetcher.normalise_listings(PAYLOAD, "bangalore", "2026-01-01T00:00:00+00:00")
    assert sorted(df["property_id"]) == ["99acres-A1", "99acres-A2"]
    a2 = df.set_index("property_id").loc["99acres-A2"]
    assert a2["area_sqft"] == 1800  # 200 sq yards
    assert a2["price_lakh"] == 160  # midpoint of 150 and 170 lakh
    assert a2["locality"] == "Jp Nagar"
    assert sum(rejected.values()) == 3


def test_refresh_without_key_is_graceful(live_dir, monkeypatch):
    monkeypatch.setattr(config, "SCRAPPA_API_KEY", "")
    result = data_fetcher.refresh()
    assert result["ok"] is False and "not configured" in result["message"]
    assert data_fetcher.data_state()["label"] == "unavailable"


def test_refresh_success_then_cache(live_dir, api_key):
    session = FakeSession([FakeResponse(200, PAYLOAD)])
    result = data_fetcher.refresh(session=session)
    assert result["ok"] and result["added"] == 2
    assert len(data_fetcher.load_live_listings()) == 2
    assert data_fetcher.data_state()["label"] == "live"
    # Second call within the minimum interval must not hit the API.
    again = data_fetcher.refresh(session=session)
    assert again.get("skipped") and session.calls == 1


def test_refresh_deduplicates_by_id(live_dir, api_key):
    data_fetcher.refresh(session=FakeSession([FakeResponse(200, PAYLOAD)]))
    result = data_fetcher.refresh(force=True, session=FakeSession([FakeResponse(200, PAYLOAD)]))
    assert result["added"] == 0 and result["updated"] == 2
    assert len(data_fetcher.load_live_listings()) == 2


def test_api_failure_keeps_previous_data(live_dir, api_key):
    data_fetcher.refresh(session=FakeSession([FakeResponse(200, PAYLOAD)]))
    failing = FakeSession([FakeResponse(503), requests.Timeout()])
    result = data_fetcher.refresh(force=True, session=failing)
    assert result["ok"] is False and "keeping previous data" in result["message"]
    assert failing.calls == 2  # retried
    assert len(data_fetcher.load_live_listings()) == 2
    status = data_fetcher.read_status()
    assert status["last_error"] and status["last_success"]


def test_cached_label_when_old(live_dir, api_key, monkeypatch):
    data_fetcher.refresh(session=FakeSession([FakeResponse(200, PAYLOAD)]))
    monkeypatch.setattr(config, "LIVE_FRESHNESS_HOURS", 0)
    assert data_fetcher.data_state()["label"] == "cached"


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_client_errors_not_retried(live_dir, api_key, status):
    session = FakeSession([FakeResponse(status)])
    result = data_fetcher.refresh(force=True, session=session)
    assert result["ok"] is False and session.calls == 1


def test_invalid_json(live_dir, api_key):
    result = data_fetcher.refresh(force=True, session=FakeSession([FakeResponse(200, None)]))
    assert result["ok"] is False and "invalid JSON" in result["message"]
