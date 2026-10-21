import math

import pandas as pd
import pytest

import housing_data


@pytest.mark.parametrize("raw, expected", [
    ("1200", 1200.0),
    ("1133 - 1384", 1258.5),
    ("34.46Sq. Meter", 34.46 * 10.7639),
    ("100Sq. Yards", 900.0),
    (1500, 1500.0),
])
def test_parse_area(raw, expected):
    assert housing_data.parse_area_sqft(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", None, "abc", "12 parsecs", "1200 - abc"])
def test_parse_area_invalid(raw):
    assert math.isnan(housing_data.parse_area_sqft(raw))


@pytest.mark.parametrize("raw, expected", [("2 BHK", 2), ("4 Bedroom", 4), ("1 RK", 1)])
def test_parse_bedrooms(raw, expected):
    assert housing_data.parse_bedrooms(raw) == expected


def _row(**overrides):
    row = dict(property_id="x", city="Bengaluru", locality="Whitefield", area_sqft=1200.0, bedrooms=2,
               bathrooms=2, balconies=1, area_type="Super built-up Area", ready_to_move=1, price_lakh=60.0,
               price_per_sqft=None, source="t", source_url="", retrieved_at="now")
    row.update(overrides)
    return row


def test_validate_rejects_bad_prices_and_areas():
    df = pd.DataFrame([
        _row(property_id="ok"),
        _row(property_id="neg-price", price_lakh=-5),
        _row(property_id="no-price", price_lakh=None),
        _row(property_id="tiny", area_sqft=100),
        _row(property_id="cramped", area_sqft=600, bedrooms=4),
        _row(property_id="silly-rate", price_lakh=4000, area_sqft=1000),
        _row(property_id="no-locality", locality=""),
    ])
    valid, reasons = housing_data.validate(df)
    assert valid["property_id"].tolist() == ["ok"]
    assert valid["price_per_sqft"].iloc[0] == 5000
    assert sum(reasons.values()) == 6


def test_training_dataset_loads_and_is_clean():
    df, report = housing_data.load_training_dataset()
    assert report["rows_raw"] == 13320
    assert report["rows_clean"] == len(df) > 10_000
    assert list(df.columns) == housing_data.SCHEMA
    assert df["property_id"].is_unique
    assert df[["area_sqft", "bedrooms", "bathrooms", "price_lakh"]].notna().all().all()
    assert df["area_sqft"].between(housing_data.MIN_AREA_SQFT, housing_data.MAX_AREA_SQFT).all()
