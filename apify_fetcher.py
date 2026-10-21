"""Fetch current Mumbai property listings from Housing.com through Apify.

Uses the public Apify actor "Housing.com Scraper"
(https://apify.com/themineworks/housing-com-scraper) with YOUR Apify token,
read from the APIFY_TOKEN environment variable. The token is never sent to
the browser and never written to any file by this script.

Apify API calls (https://docs.apify.com/api/v2):

    POST /v2/acts/<actor>/runs?waitForFinish=60      start a run with the input below
    GET  /v2/actor-runs/<run id>?waitForFinish=60     wait until it finishes
    GET  /v2/datasets/<dataset id>/items?clean=true   download the listings

    Input: {"city": "Mumbai", "listingType": "buy", "maxResults": 1000}
    Header: Authorization: Bearer <APIFY_TOKEN>

Cost at the time of writing: about US$0.001 per listing plus US$0.005 per
run, paid from Apify credit. The free Apify plan includes US$5 of credit a
month, so 1,000 listings cost roughly US$1.

The actor does not publish a formal output schema, so every field is read
defensively and records that cannot be parsed are rejected, never guessed.

Run from the command line:

    python apify_fetcher.py                 # Mumbai, respects MIN_REFRESH_INTERVAL_MINUTES
    python apify_fetcher.py --max 500       # fewer listings (cheaper)
    python apify_fetcher.py --force         # ignore the interval
    python apify_fetcher.py --from-cache    # re-read the last saved response (free)

Then train the Mumbai model:  python train_model.py --city mumbai
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import config
import housing_data

SOURCE_NAME = "Housing.com listing (via Apify)"
STATUS_FILE = "fetch_status.json"
RAW_CACHE_DIR = "raw_cache"
# How the actor names each of our cities.
ACTOR_CITY = {"mumbai": "Mumbai"}
# Address parts that are not a locality.
NOT_LOCALITY = {"mumbai", "maharashtra", "india", "mumbai suburban", "mumbai city"}
FAILED_STATES = {"FAILED", "ABORTED", "TIMED-OUT", "TIMING-OUT", "ABORTING"}


class FetchError(Exception):
    """A failure that should be reported to the user, never a stack trace."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _paths(city: str):
    data_path = config.CITIES[city]["data_path"]
    folder = data_path.parent
    return data_path, folder / STATUS_FILE, folder / RAW_CACHE_DIR


def read_status(city: str = "mumbai") -> dict:
    _, status_path, _ = _paths(city)
    try:
        with open(status_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_status(city: str, **updates) -> dict:
    _, status_path, _ = _paths(city)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = read_status(city)
    status.update(updates)
    tmp = status_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.replace(status_path)
    return status


# ---------------------------------------------------------------------------
# Parsing one listing
# ---------------------------------------------------------------------------
def _number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) and value > 0 else None
    text = str(value).replace(",", "").strip()
    match = re.search(r"[0-9]*\.?[0-9]+", text)
    return float(match.group()) if match else None


def parse_price_rupees(value) -> float | None:
    """1.25e7, "1.25 Cr", "₹85 L", "85 Lac", "₹1,20,00,000" -> rupees."""
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value) if np.isfinite(value) and value > 0 else None
    text = str(value or "").lower().replace(",", "").replace("₹", "").strip()
    number = _number(text)
    if number is None:
        return None
    if re.search(r"\b(cr|crore|crores)\b", text):
        return number * 10_000_000
    if re.search(r"\b(l|lac|lacs|lakh|lakhs)\b", text):
        return number * 100_000
    if re.search(r"\b(k|thousand)\b", text):
        return number * 1_000
    return number


def locality_from_address(address, city_name: str) -> str:
    """'Flat in Lodha Park, Worli, Mumbai, Maharashtra 400018' -> 'Worli'.

    Housing.com addresses end with the locality followed by the city, state
    and PIN code. Take the last part that is not one of those.
    """
    parts = [p.strip() for p in str(address or "").split(",")]
    skip = NOT_LOCALITY | {city_name.lower()}
    for part in reversed(parts):
        cleaned = re.sub(r"\b\d{6}\b", "", part).strip(" -")
        if cleaned and cleaned.lower() not in skip and not cleaned.isdigit():
            return housing_data.normalise_locality(cleaned)
    return ""


def _https_url(value) -> str:
    """Only plain https links are kept (shown as <img> sources on the page)."""
    text = str(value or "").strip()
    return text if re.fullmatch(r"https://[^\s\"'<>]+", text) else ""


PHOTO_FIELDS = ("photo_url", "image", "image_url", "thumbnail", "thumbnail_url", "cover_image",
                "photos", "images")


def _photo_url(item: dict) -> str:
    """First usable https image link; the field name and shape vary between scraper versions."""
    for field in PHOTO_FIELDS:
        value = item.get(field)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            value = value.get("url") or value.get("src")
        url = _https_url(value)
        if url:
            return url
    return ""


def _is_rent(item: dict) -> bool:
    kind = str(item.get("listing_type") or "").lower()
    unit = str(item.get("price_unit") or "").lower()
    label = str(item.get("price_label") or "").lower()
    return "rent" in kind or "month" in unit or "/month" in label or "per month" in label


def normalise_items(items, city: str, retrieved_at: str) -> tuple[pd.DataFrame, dict]:
    """Turn the actor's dataset items into SCHEMA rows. Returns (valid rows, rejected counts)."""
    if not isinstance(items, list):
        raise FetchError("Apify returned something other than a list of listings")
    city_name = config.CITIES[city]["name"]
    rows, rejected = [], {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for item in items:
        if not isinstance(item, dict):
            reject("not a listing")
            continue
        listing_id = item.get("listing_id") or item.get("id")
        if not listing_id:
            reject("missing listing id")
            continue
        if _is_rent(item):
            reject("rental, not for sale")
            continue
        kind = str(item.get("listing_type") or "").lower()
        if "commercial" in kind or "commercial" in str(item.get("property_type") or "").lower():
            reject("commercial property")
            continue
        price = parse_price_rupees(item.get("price"))
        if price is None:
            price = parse_price_rupees(item.get("price_label"))
        area = housing_data.parse_area_sqft(item.get("area_sqft"))
        area_type = "Built-up Area"
        if not np.isfinite(area):
            area = housing_data.parse_area_sqft(item.get("carpet_area_sqft"))
            area_type = "Carpet Area"
        if price is None:
            reject("missing price")
            continue
        if not np.isfinite(area):
            reject("missing or unparseable area")
            continue
        bedrooms = housing_data.parse_bedrooms(item.get("bhk"))
        if np.isnan(bedrooms):  # fall back to "2 BHK Flat for sale in ..." in the title
            match = re.search(r"(\d+)\s*bhk", str(item.get("title") or ""), re.I)
            bedrooms = float(match.group(1)) if match else np.nan
        locality = housing_data.normalise_locality(item.get("locality")) or locality_from_address(
            item.get("address"), city_name)
        scraped = str(item.get("scraped_at") or "").strip()
        try:
            stamp = datetime.fromisoformat(scraped.replace("Z", "+00:00")).isoformat(timespec="seconds")
        except ValueError:
            stamp = retrieved_at
        rows.append({
            "property_id": f"housing-{listing_id}",
            "city": city_name,
            "locality": locality,
            "area_sqft": round(float(area), 1),
            "bedrooms": bedrooms,
            "bathrooms": np.nan,  # not provided by the actor
            "balconies": np.nan,
            "area_type": area_type,
            "ready_to_move": np.nan,
            "property_type": str(item.get("property_type") or "Other").strip().title() or "Other",
            "price_lakh": round(price / 100_000, 2),
            "price_per_sqft": np.nan,
            "source": SOURCE_NAME,
            "source_url": str(item.get("listing_url") or item.get("url") or ""),
            "photo_url": _photo_url(item),
            "retrieved_at": stamp,
        })

    df = pd.DataFrame(rows, columns=housing_data.SCHEMA)
    if df.empty:
        return df, rejected
    df, reasons = housing_data.validate_for_city(df, city)
    for reason, count in reasons.items():
        rejected[reason] = rejected.get(reason, 0) + count
    return df.reset_index(drop=True), rejected


# ---------------------------------------------------------------------------
# Apify HTTP calls with retries
# ---------------------------------------------------------------------------
def _request(method: str, path: str, session=None, **kwargs):
    session = session or requests
    url = f"{config.APIFY_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {config.APIFY_TOKEN}", "Accept": "application/json"}
    last_error = "unknown error"
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            # waitForFinish holds the connection up to 60 s, so allow longer than that.
            resp = session.request(method, url, headers=headers, timeout=90, **kwargs)
        except requests.Timeout:
            last_error = "request timed out"
        except requests.RequestException as exc:
            last_error = f"network error ({exc.__class__.__name__})"
        else:
            if resp.status_code in (200, 201):
                try:
                    return resp.json()
                except ValueError:
                    raise FetchError("Apify returned invalid JSON")
            if resp.status_code == 401:
                raise FetchError("Apify token rejected (HTTP 401). Check APIFY_TOKEN.")
            if resp.status_code == 402:
                raise FetchError("Not enough Apify credit to run the scraper (HTTP 402).")
            if resp.status_code == 403:
                raise FetchError("Apify refused access (HTTP 403). The token may lack permission for this actor.")
            if resp.status_code == 404:
                raise FetchError(f"Apify actor or run not found (HTTP 404): {config.APIFY_ACTOR}")
            if resp.status_code == 400:
                raise FetchError("Apify rejected the scraper input (HTTP 400)")
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code} from Apify"
            else:
                raise FetchError(f"Unexpected HTTP {resp.status_code} from Apify")
        if attempt < config.MAX_RETRIES:
            time.sleep(2 ** attempt)
    raise FetchError(f"Apify unavailable after {config.MAX_RETRIES} attempts: {last_error}")


def run_actor(city: str, max_results: int, session=None) -> tuple[list, dict]:
    """Start the scraper, wait for it to finish and return (items, run info)."""
    actor_input = {"city": ACTOR_CITY[city], "listingType": "buy", "maxResults": int(max_results)}
    run = _request("POST", f"acts/{config.APIFY_ACTOR}/runs", session=session,
                   params={"waitForFinish": 60}, json=actor_input).get("data") or {}
    deadline = time.monotonic() + config.APIFY_TIMEOUT_SECONDS
    while run.get("status") not in ("SUCCEEDED", *FAILED_STATES):
        if not run.get("id"):
            raise FetchError("Apify did not return a run id")
        if time.monotonic() > deadline:
            raise FetchError(f"Scraper still running after {config.APIFY_TIMEOUT_SECONDS:.0f} s; "
                             f"check run {run['id']} in the Apify console")
        run = _request("GET", f"actor-runs/{run['id']}", session=session,
                       params={"waitForFinish": 60}).get("data") or {}
    if run["status"] != "SUCCEEDED":
        raise FetchError(f"Scraper run {run.get('id')} ended with status {run['status']}")
    items = _request("GET", f"datasets/{run['defaultDatasetId']}/items", session=session,
                     params={"clean": "true", "format": "json"})
    info = {"run_id": run.get("id"), "dataset_id": run.get("defaultDatasetId"), "input": actor_input}
    return items, info


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def load_listings(city: str = "mumbai") -> pd.DataFrame:
    data_path, _, _ = _paths(city)
    if not data_path.exists():
        return pd.DataFrame(columns=housing_data.SCHEMA)
    try:
        return pd.read_csv(data_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=housing_data.SCHEMA)


def _merge_and_save(city: str, new: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    data_path, _, _ = _paths(city)
    old = load_listings(city)
    if "photo_url" not in old.columns:
        old["photo_url"] = ""
    added = int((~new["property_id"].isin(set(old["property_id"].astype(str)))).sum())
    # Newest copy of a listing wins.
    combined = pd.concat([old, new], ignore_index=True).drop_duplicates(subset="property_id", keep="last")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_path.with_suffix(".tmp")
    combined[housing_data.SCHEMA].to_csv(tmp, index=False)
    tmp.replace(data_path)
    return added, combined


def rebuild_from_cache(city: str = "mumbai") -> dict:
    """Re-read the last raw Apify response (no API call, no credit used).

    Useful after a parser change, for example to add listing photos to
    listings that were fetched before photos were kept.
    """
    _, _, raw_dir = _paths(city)
    raw_path = raw_dir / f"apify_{city}_latest.json"
    try:
        with open(raw_path) as f:
            items = json.load(f)
        stamp = read_status(city).get("last_success") or _now().isoformat(timespec="seconds")
        new, rejected = normalise_items(items, city, stamp)
    except FileNotFoundError:
        return {"ok": False, "message": f"No saved Apify response at {raw_path}. Run a normal fetch first."}
    except (json.JSONDecodeError, FetchError) as exc:
        return {"ok": False, "message": f"Could not read {raw_path}: {exc}"}
    added, combined = _merge_and_save(city, new)
    photos = int((new["photo_url"].fillna("") != "").sum())
    return {"ok": True, "rejected": rejected,
            "message": f"Rebuilt {len(new)} listings from the saved response ({photos} with a photo). "
                       f"{len(combined)} listings saved. No Apify credit used."}


def refresh(city: str = "mumbai", max_results: int | None = None, force: bool = False,
            session=None) -> dict:
    """Fetch, validate, de-duplicate and save listings for one city.

    Never raises: returns a dict with "ok" and a human message. On any
    failure the previously saved CSV is left untouched.
    """
    if city not in ACTOR_CITY:
        return {"ok": False, "message": f"Apify fetching is set up for {sorted(ACTOR_CITY)} only."}
    if not config.APIFY_TOKEN:
        return {"ok": False, "message": "APIFY_TOKEN is not set. Add it to your .env file (see .env.example)."}
    max_results = max_results or config.APIFY_MAX_RESULTS
    status = read_status(city)
    last_success = status.get("last_success")
    if last_success and not force:
        age = _now() - datetime.fromisoformat(last_success)
        if age < timedelta(minutes=config.MIN_REFRESH_INTERVAL_MINUTES):
            return {"ok": True, "skipped": True,
                    "message": f"Listings were fetched {int(age.total_seconds() // 60)} minutes ago; "
                               f"use --force to fetch again (each fetch uses Apify credit)."}

    retrieved_at = _now().isoformat(timespec="seconds")
    _write_status(city, last_attempt=retrieved_at)
    data_path, _, raw_dir = _paths(city)
    try:
        items, info = run_actor(city, max_results, session=session)
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Keep the latest raw response so a parsing problem can be inspected.
        with open(raw_dir / f"apify_{city}_latest.json", "w") as f:
            json.dump(items, f)
        new, rejected = normalise_items(items, city, retrieved_at)
    except FetchError as exc:
        _write_status(city, last_error=str(exc), last_error_at=retrieved_at)
        return {"ok": False, "message": f"Fetch failed, keeping previous data. {exc}"}
    except Exception as exc:  # never leave a half-written file
        _write_status(city, last_error=f"unexpected error: {exc.__class__.__name__}", last_error_at=retrieved_at)
        return {"ok": False, "message": f"Fetch failed unexpectedly ({exc.__class__.__name__}); keeping previous data."}

    added, combined = _merge_and_save(city, new)
    result = {"ok": True, "received": len(items), "valid": len(new), "added": added,
              "rejected": rejected, "total": len(combined), **info}
    _write_status(city, last_success=retrieved_at, last_result=result, last_error=None)
    result["message"] = (f"Received {len(items)} listings from Apify: {len(new)} valid ({added} new), "
                         f"{sum(rejected.values())} rejected. {len(combined)} {config.CITIES[city]['name']} "
                         f"listings saved to {data_path}.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=sorted(ACTOR_CITY), default="mumbai")
    parser.add_argument("--max", type=int, default=None, help=f"listings to request (default {config.APIFY_MAX_RESULTS})")
    parser.add_argument("--force", action="store_true", help="ignore MIN_REFRESH_INTERVAL_MINUTES")
    parser.add_argument("--from-cache", action="store_true",
                        help="re-read the last saved Apify response instead of calling Apify (free)")
    args = parser.parse_args()
    if args.from_cache:
        result = rebuild_from_cache(args.city)
    else:
        result = refresh(args.city, max_results=args.max, force=args.force)
    print(result["message"])
    if result.get("rejected"):
        print("Rejected:", result["rejected"])
    if result["ok"] and not result.get("skipped"):
        print(f"Next: python train_model.py --city {args.city}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
