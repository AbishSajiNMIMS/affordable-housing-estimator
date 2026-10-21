"""Mumbai: Apify fetcher, per-city training and the Mumbai page.

All listings here are made up (conftest.make_apify_items) and only test the code.
"""
import json

import pandas as pd
import pytest
import requests

import apify_fetcher
import config
from conftest import make_apify_items

RETRIEVED = "2026-10-21T06:00:00+00:00"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (12_500_000, 12_500_000),
    ("₹1.25 Cr", 12_500_000),
    ("85 L", 8_500_000),
    ("85 Lac", 8_500_000),
    ("₹1,20,00,000", 12_000_000),
    ("", None),
    (None, None),
])
def test_parse_price_rupees(value, expected):
    assert apify_fetcher.parse_price_rupees(value) == expected


@pytest.mark.parametrize("address,expected", [
    ("Lodha Park, Worli, Mumbai, Maharashtra 400018", "Worli"),
    ("Andheri West, Mumbai", "Andheri West"),
    ("powai, Mumbai, 400076", "Powai"),
    ("Mumbai, Maharashtra", ""),
    (None, ""),
])
def test_locality_from_address(address, expected):
    assert apify_fetcher.locality_from_address(address, "Mumbai") == expected


def test_normalise_items_keeps_valid_and_rejects_bad():
    good = make_apify_items(3)
    rent = dict(good[0], listing_id="R1", listing_type="rent", price=45000, price_unit="per month")
    commercial = dict(good[0], listing_id="C1", property_type="commercial office")
    no_price = dict(good[0], listing_id="P1", price=None, price_label="Price on request")
    label_only = dict(good[1], listing_id="L1", price=None, price_label="₹2.4 Cr")
    carpet = dict(good[2], listing_id="A1", area_sqft=None, carpet_area_sqft="610")
    title_bhk = dict(good[2], listing_id="B1", bhk=None, title="2 BHK Flat for sale in Powai", area_sqft=900)
    df, rejected = apify_fetcher.normalise_items(
        good + [rent, commercial, no_price, label_only, carpet, title_bhk, "junk"], "mumbai", RETRIEVED)

    assert set(df["property_id"]) == {"housing-T0", "housing-T1", "housing-T2", "housing-L1", "housing-A1",
                                      "housing-B1"}
    assert rejected == {"rental, not for sale": 1, "commercial property": 1, "missing price": 1, "not a listing": 1}
    row = df.set_index("property_id")
    assert row.loc["housing-L1", "price_lakh"] == 240
    assert row.loc["housing-A1", "area_type"] == "Carpet Area"
    assert row.loc["housing-B1", "bedrooms"] == 2
    assert (df["city"] == "Mumbai").all()
    assert df["bathrooms"].isna().all()  # the scraper has no bathroom count; never invented
    assert row.loc["housing-T0", "source_url"].startswith("https://housing.com/")
    assert row.loc["housing-T0", "retrieved_at"] == "2026-10-21T06:00:00+00:00"


def test_mumbai_price_limits_are_wider_than_bengaluru():
    # A Worli flat at Rs 90,000/sq ft is plausible in Mumbai but outside the Bengaluru limit.
    item = dict(make_apify_items(1)[0], area_sqft=1000, price=90_000_000, bhk=2)
    df, rejected = apify_fetcher.normalise_items([item], "mumbai", RETRIEVED)
    assert len(df) == 1, rejected


def test_normalise_items_rejects_non_list():
    with pytest.raises(apify_fetcher.FetchError):
        apify_fetcher.normalise_items({"error": "x"}, "mumbai", RETRIEVED)


# ---------------------------------------------------------------------------
# Apify HTTP flow with a fake session (no network, no token used)
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeApify:
    """Start run -> still RUNNING -> SUCCEEDED -> dataset items."""

    def __init__(self, items, fail_status=None):
        self.items = items
        self.fail_status = fail_status
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        self.calls.append((method, url, json, headers))
        if self.fail_status:
            return FakeResponse(self.fail_status, {"error": {"type": "x"}})
        if method == "POST" and url.endswith("/runs"):
            return FakeResponse(201, {"data": {"id": "run1", "status": "RUNNING", "defaultDatasetId": "ds1"}})
        if "/actor-runs/run1" in url:
            return FakeResponse(200, {"data": {"id": "run1", "status": "SUCCEEDED", "defaultDatasetId": "ds1"}})
        if "/datasets/ds1/items" in url:
            return FakeResponse(200, self.items)
        return FakeResponse(404, {})


@pytest.fixture()
def token(monkeypatch):
    monkeypatch.setattr(config, "APIFY_TOKEN", "test-token-not-real")


def test_refresh_without_token_does_not_call_api(mumbai_dir, monkeypatch):
    monkeypatch.setattr(config, "APIFY_TOKEN", "")
    session = FakeApify(make_apify_items(5))
    result = apify_fetcher.refresh("mumbai", session=session)
    assert result["ok"] is False and "APIFY_TOKEN" in result["message"]
    assert session.calls == []


def test_refresh_saves_listings_and_status(mumbai_dir, token):
    session = FakeApify(make_apify_items(20))
    result = apify_fetcher.refresh("mumbai", max_results=20, session=session)
    assert result["ok"], result
    assert result["valid"] == 20 and result["added"] == 20

    method, url, body, headers = session.calls[0]
    assert method == "POST" and url.endswith("/acts/themineworks~housing-com-scraper/runs")
    assert body == {"city": "Mumbai", "listingType": "buy", "maxResults": 20}
    assert headers["Authorization"] == "Bearer test-token-not-real"

    saved = pd.read_csv(mumbai_dir / "housing_com_mumbai.csv")
    assert len(saved) == 20
    status = json.loads((mumbai_dir / "fetch_status.json").read_text())
    assert status["last_success"] and status["last_error"] is None
    # The token must never be written to disk.
    for path in mumbai_dir.rglob("*"):
        if path.is_file():
            assert "test-token-not-real" not in path.read_text()


def test_refresh_respects_interval_and_merges(mumbai_dir, token):
    apify_fetcher.refresh("mumbai", session=FakeApify(make_apify_items(10)))
    skipped = apify_fetcher.refresh("mumbai", session=FakeApify(make_apify_items(10)))
    assert skipped.get("skipped")
    # Forced: 10 listings again (same ids) plus 5 new ones -> 15 stored.
    more = make_apify_items(10) + [dict(i, listing_id=f"N{n}") for n, i in enumerate(make_apify_items(5, seed=1))]
    result = apify_fetcher.refresh("mumbai", force=True, session=FakeApify(more))
    assert result["added"] == 5 and result["total"] == 15


def test_refresh_failure_keeps_previous_data(mumbai_dir, token, monkeypatch):
    monkeypatch.setattr(apify_fetcher.time, "sleep", lambda s: None)
    apify_fetcher.refresh("mumbai", session=FakeApify(make_apify_items(10)))
    for status in (401, 402, 503):
        result = apify_fetcher.refresh("mumbai", force=True, session=FakeApify([], fail_status=status))
        assert result["ok"] is False and "keeping previous data" in result["message"]
    assert len(pd.read_csv(mumbai_dir / "housing_com_mumbai.csv")) == 10
    assert "503" in apify_fetcher.read_status("mumbai")["last_error"]


def test_refresh_network_error(mumbai_dir, token, monkeypatch):
    monkeypatch.setattr(apify_fetcher.time, "sleep", lambda s: None)

    class Broken:
        def request(self, *a, **k):
            raise requests.ConnectionError("down")

    result = apify_fetcher.refresh("mumbai", session=Broken())
    assert result["ok"] is False and "network error" in result["message"]


# ---------------------------------------------------------------------------
# Training and estimating
# ---------------------------------------------------------------------------
def test_training_needs_enough_rows(tmp_path):
    import train_model

    df, _ = apify_fetcher.normalise_items(make_apify_items(30), "mumbai", RETRIEVED)
    path = tmp_path / "small.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="need at least"):
        train_model.train(city="mumbai", epochs=1, artifacts_dir=tmp_path / "out", data_path=path, verbose=0)


def test_mumbai_estimate(mumbai_artifacts):
    from estimator import Estimator

    est = Estimator("mumbai", mumbai_artifacts)
    assert est.features["numeric"] == ["bedrooms"]
    assert "Apartment" in est.property_types
    result = est.estimate({"locality": "Powai", "area_sqft": 900, "bedrooms": 2, "property_type": "Apartment"})
    assert result["price_lakh"] > 0
    assert len(result["similar"]) == 3
    assert all(s["bathrooms"] is None for s in result["similar"])
    assert all(s["source_url"].startswith("https://housing.com/") for s in result["similar"])
    assert est.metrics["dataset"]["city"] == "Mumbai"


def test_mumbai_page_and_estimate(client, mumbai_artifacts, mumbai_dir):
    page = client.get("/?city=mumbai")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Property type" in html and 'name="bathrooms"' not in html
    assert "Housing.com" in html

    resp = client.post("/", data={"city": "mumbai", "locality": "Powai", "area_sqft": "900",
                                  "bedrooms": "2", "property_type": "Apartment"})
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Estimated price" in html and "https://housing.com/" in html

    api = client.post("/api/estimate", json={"city": "mumbai", "locality": "Powai", "area_sqft": 900,
                                             "bedrooms": 2, "property_type": "Apartment"})
    assert api.status_code == 200 and api.get_json()["city"] == "mumbai"

    bad = client.post("/", data={"city": "mumbai", "locality": "Powai", "area_sqft": "900",
                                 "bedrooms": "2", "property_type": "Castle"})
    assert bad.status_code == 400


def test_mumbai_not_ready_page(client, mumbai_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "empty")
    resp = client.get("/?city=mumbai")
    assert resp.status_code == 503
    html = resp.get_data(as_text=True)
    assert "not set up on this server yet" in html and "apify_fetcher.py" in html


def test_bengaluru_page_has_city_switch(client):
    html = client.get("/").get_data(as_text=True)
    assert 'id="city-select"' in html and ">Mumbai</option>" in html
    assert 'name="bathrooms"' in html


def test_unknown_city_falls_back_to_bengaluru(client):
    resp = client.get("/?city=atlantis")
    assert resp.status_code == 200 and "Bengaluru" in resp.get_data(as_text=True)


def test_bengaluru_not_ready_keeps_city_selector(client, mumbai_artifacts, mumbai_dir, tmp_path, monkeypatch):
    # Only Mumbai trained: the home page explains Bengaluru and still offers Mumbai.
    empty = tmp_path / "only-mumbai"
    import shutil
    shutil.copytree(mumbai_artifacts, empty / "mumbai")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", empty)
    resp = client.get("/")
    assert resp.status_code == 503
    html = resp.get_data(as_text=True)
    assert "train_model.py --all" in html and 'id="city-select"' in html
    assert client.get("/?city=mumbai").status_code == 200


def test_legacy_bengaluru_artifacts_still_load(artifacts_dir, tmp_path, monkeypatch):
    # Models trained before per-city folders lived directly in artifacts/.
    import shutil
    from estimator import Estimator
    shutil.copytree(artifacts_dir, tmp_path / "old")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "old")
    assert Estimator("bengaluru").dir == str(tmp_path / "old")


# ---------------------------------------------------------------------------
# Photos and area maps
# ---------------------------------------------------------------------------
def test_photo_urls_kept_only_when_https():
    items = make_apify_items(3)
    items[0]["photo_url"] = "javascript:alert(1)"
    items[1]["photo_url"] = "https://img.example/x.jpg"
    df, _ = apify_fetcher.normalise_items(items, "mumbai", RETRIEVED)
    photos = df.set_index("property_id")["photo_url"]
    assert photos["housing-T0"] == "" and photos["housing-T1"] == "https://img.example/x.jpg"


def test_rebuild_from_cache_adds_photos_without_api(mumbai_dir, token):
    apify_fetcher.refresh("mumbai", session=FakeApify(make_apify_items(10)))
    # Simulate a CSV saved by the older version, which had no photo column.
    path = mumbai_dir / "housing_com_mumbai.csv"
    pd.read_csv(path).drop(columns=["photo_url"]).to_csv(path, index=False)
    result = apify_fetcher.rebuild_from_cache("mumbai")
    assert result["ok"] and "No Apify credit used" in result["message"]
    assert (pd.read_csv(path)["photo_url"].fillna("") != "").sum() > 0


def test_mumbai_cards_show_listing_photos(client, mumbai_artifacts, mumbai_dir):
    html = client.post("/", data={"city": "mumbai", "locality": "Powai", "area_sqft": "900", "bedrooms": "2",
                                  "property_type": "Apartment"}).get_data(as_text=True)
    assert "housingcdn.com/test/" in html and "Listing photo" in html


def test_area_map_needs_key_and_known_locality(client, monkeypatch, tmp_path):
    import area_maps

    monkeypatch.setattr(config, "GOOGLE_MAPS_API_KEY", "")
    assert client.get("/area-map?city=bengaluru&locality=Whitefield").status_code == 404

    calls = []

    class FakeGoogle:
        def get(self, url, params=None, timeout=None):
            calls.append(params)
            return type("R", (), {"status_code": 200, "headers": {"Content-Type": "image/png"},
                                  "content": b"\x89PNG fake"})()

    monkeypatch.setattr(config, "GOOGLE_MAPS_API_KEY", "test-maps-key")
    monkeypatch.setattr(area_maps, "requests", FakeGoogle())
    ok = client.get("/area-map?city=bengaluru&locality=whitefield")
    assert ok.status_code == 200 and ok.data == b"\x89PNG fake"
    assert calls[0]["center"] == "Whitefield, Bengaluru, India"
    client.get("/area-map?city=bengaluru&locality=Whitefield")
    assert len(calls) == 1  # second request served from the cache
    assert client.get("/area-map?city=bengaluru&locality=Somewhere%20Else").status_code == 404
    # The key never reaches the page.
    page = client.post("/", data={"locality": "Whitefield", "area_sqft": "1200", "bedrooms": "2",
                                  "bathrooms": "2", "balconies": "1"}).get_data(as_text=True)
    assert "/area-map?city=bengaluru" in page and "test-maps-key" not in page


def test_photo_from_list_or_nested_field():
    a, b = make_apify_items(2)
    a.pop("photo_url"); a["photos"] = [{"url": "https://img.example/a.jpg"}]
    b.pop("photo_url"); b["images"] = ["https://img.example/b.jpg"]
    df, _ = apify_fetcher.normalise_items([a, b], "mumbai", RETRIEVED)
    assert list(df["photo_url"]) == ["https://img.example/a.jpg", "https://img.example/b.jpg"]
