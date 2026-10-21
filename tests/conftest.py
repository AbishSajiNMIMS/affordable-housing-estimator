"""Shared test fixtures.

Small models are trained once per test run into a temporary folder, so the
tests never touch your real artifacts/, data/live/ or data/mumbai/ folders:

* Bengaluru: 3,000 rows of the real dataset, 5 epochs.
* Mumbai: made-up listings shaped like the Apify Housing.com scraper output
  (see make_apify_items). They exist only to exercise the code and are
  never shown on the real website.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

MUMBAI_LOCALITIES = ["Andheri West", "Powai", "Borivali West", "Chembur", "Malad West", "Worli"]
MUMBAI_RATES = {"Andheri West": 30000, "Powai": 26000, "Borivali West": 21000, "Chembur": 24000,
                "Malad West": 22000, "Worli": 60000}


def make_apify_items(n=400, seed=0):
    """Fake items with the field names of the Apify Housing.com scraper. Test use only."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        loc = MUMBAI_LOCALITIES[i % len(MUMBAI_LOCALITIES)]
        bhk = int(rng.integers(1, 4))
        area = float(round(bhk * 380 + rng.normal(0, 60)))
        ptype = "apartment" if i % 7 else "independent house"
        price = area * MUMBAI_RATES[loc] * float(rng.uniform(0.85, 1.15))
        items.append({
            "listing_id": f"T{i}",
            "title": f"{bhk} BHK Flat for sale in {loc}",
            "listing_type": "buy",
            "property_type": ptype,
            "bhk": bhk,
            "price": round(price),
            "price_label": f"₹{price / 1e7:.2f} Cr",
            "area_sqft": area,
            "address": f"Test Tower {i}, {loc}, Mumbai, Maharashtra 400001",
            "city": "Mumbai",
            "listing_url": f"https://housing.com/in/buy/resale/page/test-{i}",
            "scraped_at": "2026-10-21T06:00:00Z",
        })
    return items


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    import train_model

    path = tmp_path_factory.mktemp("artifacts")
    original = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = path
    train_model.train(city="bengaluru", epochs=5, sample=3000, artifacts_dir=path / "bengaluru", verbose=0)
    yield path / "bengaluru"
    config.ARTIFACTS_DIR = original


@pytest.fixture(scope="session")
def mumbai_artifacts(artifacts_dir, tmp_path_factory):
    """Mumbai model trained on the fake Apify-shaped listings."""
    import apify_fetcher
    import train_model

    df, _ = apify_fetcher.normalise_items(make_apify_items(), "mumbai", "2026-10-21T06:00:00+00:00")
    data_path = tmp_path_factory.mktemp("mumbai") / "housing_com_mumbai.csv"
    df.to_csv(data_path, index=False)
    out = config.ARTIFACTS_DIR / "mumbai"
    train_model.train(city="mumbai", epochs=5, artifacts_dir=out, data_path=data_path, verbose=0)
    return out


@pytest.fixture()
def live_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path / "live")
    return tmp_path / "live"


@pytest.fixture()
def mumbai_dir(tmp_path, monkeypatch):
    """Point the Mumbai data file at an empty temporary folder."""
    spec = dict(config.CITIES["mumbai"], data_path=tmp_path / "mumbai" / "housing_com_mumbai.csv")
    monkeypatch.setitem(config.CITIES, "mumbai", spec)
    return tmp_path / "mumbai"


@pytest.fixture(scope="session")
def estimator(artifacts_dir):
    from estimator import Estimator

    return Estimator("bengaluru", artifacts_dir)


@pytest.fixture()
def client(artifacts_dir, live_dir):
    import app as app_module

    app_module._estimators.clear()
    app_module._estimator_errors.clear()
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
    app_module._estimators.clear()
    app_module._estimator_errors.clear()
