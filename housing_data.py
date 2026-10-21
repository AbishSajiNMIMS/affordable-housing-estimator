"""Load, parse, validate and clean the property data.

The same validation rules are used for the historical training dataset and
for live listings fetched from the API, so both end up in one schema:

    property_id, city, locality, area_sqft, bedrooms, bathrooms, balconies,
    area_type, ready_to_move, property_type, price_lakh, price_per_sqft,
    source, source_url, retrieved_at

Columns a source does not report are left empty (for example Mumbai
listings have no bathroom or balcony counts).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config

SCHEMA = [
    "property_id",
    "city",
    "locality",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "balconies",
    "area_type",
    "ready_to_move",
    "property_type",
    "price_lakh",
    "price_per_sqft",
    "source",
    "source_url",
    "retrieved_at",
]

# Validation limits. They remove obvious data-entry errors, not "expensive"
# or "cheap" homes. 1 lakh = 100,000 rupees.
MIN_AREA_SQFT, MAX_AREA_SQFT = 250, 30_000
MIN_PRICE_LAKH, MAX_PRICE_LAKH = 5, 5_000
MIN_PPSF, MAX_PPSF = 1_500, 50_000  # rupees per square foot
MIN_SQFT_PER_BEDROOM = 250
MAX_BEDROOMS = 10

AREA_TYPES = ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"]

# Unit conversions to square feet for area values such as "34.46Sq. Meter".
UNIT_TO_SQFT = {
    "sq. meter": 10.7639,
    "sq. yards": 9.0,
    "sq.yards": 9.0,
    "sq.yard": 9.0,
    "sq. yard": 9.0,
    "sq.m": 10.7639,
    "sq.ft": 1.0,
    "sq. ft": 1.0,
    "sqft": 1.0,
    "sq.ft.": 1.0,
    "acres": 43_560.0,
    "cents": 435.6,
    "guntha": 1_089.0,
    "grounds": 2_400.0,
}


def parse_area_sqft(value) -> float:
    """Convert raw area text to square feet.

    Handles plain numbers ("1200"), ranges ("1133 - 1384" -> midpoint) and
    values with units ("34.46Sq. Meter"). Returns NaN when it cannot parse.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else np.nan
    text = str(value).strip().lower().replace(",", "")
    if not text:
        return np.nan
    if " - " in text:
        parts = [parse_area_sqft(p) for p in text.split(" - ")]
        return float(np.mean(parts)) if all(np.isfinite(parts)) else np.nan
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-z. ]+)", text)
    if match:
        number, unit = float(match.group(1)), match.group(2).strip()
        factor = UNIT_TO_SQFT.get(unit)
        if factor:
            return number * factor
    return np.nan


def parse_bedrooms(value) -> float:
    """'2 BHK', '4 Bedroom', '1 RK' -> 2, 4, 1."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.match(r"\s*(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def normalise_area_type(value) -> str:
    """'Super built-up  Area' (double space in the raw file) -> 'Super built-up Area'."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if text in AREA_TYPES else "Super built-up Area"


def normalise_locality(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.title() if text else ""


def validate(df: pd.DataFrame, max_price_lakh: float = MAX_PRICE_LAKH,
             max_ppsf: float = MAX_PPSF,
             min_sqft_per_bedroom: float = MIN_SQFT_PER_BEDROOM) -> tuple[pd.DataFrame, dict]:
    """Apply the validation rules to a frame already in SCHEMA form.

    The upper price limits can be raised per city (Mumbai is far more
    expensive per sq ft than Bengaluru). Returns the valid rows and a dict
    counting why rows were rejected.
    """
    reasons = {}

    def reject(mask: pd.Series, reason: str):
        nonlocal df
        count = int(mask.sum())
        if count:
            reasons[reason] = reasons.get(reason, 0) + count
        df = df[~mask]

    reject(df["locality"].fillna("").eq(""), "missing locality")
    reject(df["area_sqft"].isna(), "missing or unparseable area")
    reject(df["price_lakh"].isna(), "missing price")
    reject(df["bedrooms"].isna(), "missing bedrooms")
    reject(~df["area_sqft"].between(MIN_AREA_SQFT, MAX_AREA_SQFT), "area out of range")
    reject(~df["price_lakh"].between(MIN_PRICE_LAKH, max_price_lakh), "price out of range")
    reject(~df["bedrooms"].between(1, MAX_BEDROOMS), "bedrooms out of range")
    reject(df["area_sqft"] / df["bedrooms"] < min_sqft_per_bedroom, "too small for bedroom count")
    reject(df["bathrooms"].notna() & (df["bathrooms"] > df["bedrooms"] + 2), "implausible bathroom count")
    ppsf = df["price_lakh"] * 100_000 / df["area_sqft"]
    reject(~ppsf.between(MIN_PPSF, max_ppsf), "price per sq ft out of range")

    df = df.copy()
    df["price_per_sqft"] = (df["price_lakh"] * 100_000 / df["area_sqft"]).round(0)
    return df, reasons


def validate_for_city(df: pd.DataFrame, city: str) -> tuple[pd.DataFrame, dict]:
    spec = config.CITIES[city]
    return validate(df, spec["max_price_lakh"], spec["max_price_per_sqft"], spec["min_sqft_per_bedroom"])


def load_training_dataset(path=None, city: str = "bengaluru") -> tuple[pd.DataFrame, dict]:
    """Return (clean frame, report) for one city's training data."""
    if city == "mumbai":
        return load_listings_dataset(path or config.CITIES["mumbai"]["data_path"], "mumbai")
    if city != "bengaluru":
        raise ValueError(f"Unknown city {city!r}. Choose from {sorted(config.CITIES)}")
    path = path or config.RAW_DATASET_PATH
    raw = pd.read_csv(path)
    required = {"area_type", "availability", "location", "size", "total_sqft", "bath", "balcony", "price"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            # Row number in the original file makes a stable identifier.
            "property_id": ["blr-" + str(i + 2) for i in range(len(raw))],
            "city": config.TRAINING_CITY,
            "locality": raw["location"].map(normalise_locality),
            "area_sqft": raw["total_sqft"].map(parse_area_sqft),
            "bedrooms": raw["size"].map(parse_bedrooms),
            "bathrooms": pd.to_numeric(raw["bath"], errors="coerce"),
            "balconies": pd.to_numeric(raw["balcony"], errors="coerce"),
            "area_type": raw["area_type"].map(normalise_area_type),
            "ready_to_move": raw["availability"].astype(str).str.strip().eq("Ready To Move").astype(int),
            "property_type": "",  # not recorded in this dataset
            "price_lakh": pd.to_numeric(raw["price"], errors="coerce"),
            "source": config.TRAINING_SOURCE_NAME,
            "source_url": config.TRAINING_SOURCE_URL,
            "retrieved_at": "historical (collection date not documented; ~2017-18)",
        }
    )
    rows_raw = len(df)

    # The raw file repeats some listings. Treat rows with the same
    # characteristics and price as one property.
    dedupe_cols = ["locality", "area_sqft", "bedrooms", "bathrooms", "balconies", "area_type", "price_lakh"]
    df = df.drop_duplicates(subset=dedupe_cols)
    duplicates = rows_raw - len(df)

    df, reasons = validate(df)

    # Missing bathrooms/balconies: fill with the median for that bedroom count.
    for col in ["bathrooms", "balconies"]:
        df[col] = df[col].fillna(df.groupby("bedrooms")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    df = df[SCHEMA].reset_index(drop=True)
    report = {
        "rows_raw": rows_raw,
        "duplicates_removed": duplicates,
        "rows_rejected": reasons,
        "rows_clean": len(df),
        "localities": int(df["locality"].nunique()),
    }
    return df, report


def load_listings_dataset(path, city: str) -> tuple[pd.DataFrame, dict]:
    """Read listings saved by apify_fetcher.py (already in SCHEMA form).

    The fetcher validated them once; they are validated again here so a
    hand-edited or older file cannot slip bad rows into training.
    """
    spec = config.CITIES[city]
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No training data for {spec['name']} at {path}. Get it with: {spec['how_to_get_data']}"
        ) from None
    missing = set(SCHEMA) - set(raw.columns) - {"price_per_sqft"}
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    rows_raw = len(raw)
    df = raw.drop_duplicates(subset=["property_id"])
    duplicates = rows_raw - len(df)
    df["property_type"] = df["property_type"].fillna("Other").astype(str)
    df, reasons = validate_for_city(df, city)
    df = df[SCHEMA].reset_index(drop=True)
    report = {
        "rows_raw": rows_raw,
        "duplicates_removed": duplicates,
        "rows_rejected": reasons,
        "rows_clean": len(df),
        "localities": int(df["locality"].nunique()),
        "fetched_from": str(df["retrieved_at"].min()) if len(df) else None,
        "fetched_to": str(df["retrieved_at"].max()) if len(df) else None,
    }
    return df, report
