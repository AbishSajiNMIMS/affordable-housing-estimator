"""Central configuration.

Every setting can be overridden with an environment variable, so the same
code runs on a laptop, inside Docker and on AWS ECS without edits.
"""
import os
from pathlib import Path

try:  # python-dotenv is optional: it only loads a local .env file if present
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DATASET_PATH = Path(
    os.getenv("RAW_DATASET_PATH", DATA_DIR / "raw" / "bengaluru_house_prices.csv")
)
# Live data lives in its own folder so it can be mounted on persistent
# storage (Amazon EFS on ECS) and is never committed to Git.
LIVE_DIR = Path(os.getenv("LIVE_DIR", DATA_DIR / "live"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))

# ---------------------------------------------------------------------------
# Training dataset provenance (shown on the website)
# ---------------------------------------------------------------------------
TRAINING_CITY = "Bengaluru"
TRAINING_SOURCE_NAME = "Bengaluru House Price Data (Kaggle, amitabhajoy)"
TRAINING_SOURCE_URL = "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data"
TRAINING_SOURCE_LICENSE = "CC0: Public Domain (as declared on Kaggle)"
# Public mirror used by scripts/download_dataset.py (identical columns, 13,320 rows)
TRAINING_MIRROR_URL = (
    "https://raw.githubusercontent.com/codebasics/py/master/"
    "DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv"
)
TRAINING_SHA256 = "23ddf71be7d79a259b62e125b9d56cb531f9d051b0da650c0fe7cc82365a1015"

# ---------------------------------------------------------------------------
# Cities. Each city has its own training data, its own trained models in
# artifacts/<city>/ and only the input fields its data actually contains.
# ---------------------------------------------------------------------------
MUMBAI_DATA_PATH = Path(os.getenv("MUMBAI_DATA_PATH", DATA_DIR / "mumbai" / "housing_com_mumbai.csv"))

CITIES = {
    "bengaluru": {
        "name": "Bengaluru",
        "data_path": RAW_DATASET_PATH,
        "numeric": ["bedrooms", "bathrooms", "balconies", "ready_to_move"],
        "categorical": ["locality", "area_type"],
        "knn": ["log_area", "bedrooms", "bathrooms"],
        "pca": ["log_area", "bedrooms", "bathrooms", "balconies", "log_price"],
        "source_name": TRAINING_SOURCE_NAME,
        "source_url": TRAINING_SOURCE_URL,
        "license": TRAINING_SOURCE_LICENSE,
        "price_note": "historical asking prices (roughly 2017-18), not sale prices and not today's market",
        "max_price_per_sqft": 50_000,
        "max_price_lakh": 5_000,
        "min_sqft_per_bedroom": 250,
        "how_to_get_data": "included in the repository (data/raw/)",
    },
    "mumbai": {
        "name": "Mumbai",
        "data_path": MUMBAI_DATA_PATH,
        # Housing.com listing cards give BHK but not bathrooms or balconies.
        "numeric": ["bedrooms"],
        "categorical": ["locality", "property_type"],
        "knn": ["log_area", "bedrooms"],
        "pca": ["log_area", "bedrooms", "log_price"],
        "source_name": "Housing.com listings via the Apify Housing.com scraper",
        "source_url": "https://apify.com/themineworks/housing-com-scraper",
        "license": "Not redistributable: fetched with your own Apify token, kept out of Git",
        "price_note": "current asking prices from listings fetched on the date shown, not sale prices",
        # South Mumbai flats can exceed Rs 1 lakh per sq ft.
        "max_price_per_sqft": 200_000,
        "max_price_lakh": 20_000,
        # Mumbai listings often quote carpet area: a 2 BHK can be 450 sq ft.
        "min_sqft_per_bedroom": 150,
        "how_to_get_data": "python apify_fetcher.py --city mumbai",
    },
}
DEFAULT_CITY = "bengaluru"
# Minimum valid listings before a city's model is trained.
MIN_TRAINING_ROWS = int(os.getenv("MIN_TRAINING_ROWS", "300"))

# ---------------------------------------------------------------------------
# Apify Housing.com scraper (Mumbai listings) - optional
# ---------------------------------------------------------------------------
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
APIFY_ACTOR = os.getenv("APIFY_ACTOR", "themineworks~housing-com-scraper")
APIFY_BASE_URL = os.getenv("APIFY_BASE_URL", "https://api.apify.com/v2")
# Listings per fetch. The free Apify plan ($5/month) covers about 4,990 listings a month.
APIFY_MAX_RESULTS = int(os.getenv("APIFY_MAX_RESULTS", "1000"))
APIFY_TIMEOUT_SECONDS = float(os.getenv("APIFY_TIMEOUT_SECONDS", "900"))


def city_artifacts_dir(city: str) -> Path:
    return Path(ARTIFACTS_DIR) / city


# ---------------------------------------------------------------------------
# Live listings (Scrappa 99acres API) - optional
# ---------------------------------------------------------------------------
SCRAPPA_API_KEY = os.getenv("SCRAPPA_API_KEY", "").strip()
SCRAPPA_BASE_URL = os.getenv("SCRAPPA_BASE_URL", "https://scrappa.co/api/99acres/v1")
# City token as it appears in 99acres URLs. Comma separated for several cities.
LIVE_CITIES = [c.strip() for c in os.getenv("LIVE_CITIES", "bangalore").split(",") if c.strip()]
# Do not call the API again if the last successful fetch is newer than this.
MIN_REFRESH_INTERVAL_MINUTES = int(os.getenv("MIN_REFRESH_INTERVAL_MINUTES", "360"))
# Background refresh inside the web app. 0 disables it (the default).
AUTO_REFRESH_HOURS = float(os.getenv("AUTO_REFRESH_HOURS", "0"))
# Data younger than this is labelled "live"; older successful data is "cached".
LIVE_FRESHNESS_HOURS = float(os.getenv("LIVE_FRESHNESS_HOURS", "24"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Protects the manual refresh button so visitors cannot spend API credits.
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
