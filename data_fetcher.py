"""Fetch current Indian property listings from the Scrappa 99acres API.

Documented endpoint used (https://scrappa.co/docs/99acres-api/99acres_city_listings):

    GET https://scrappa.co/api/99acres/v1/city-listings?city=<city token>
    Header: x-api-key: <your key>

Each successful request costs 1 Scrappa credit (500 free credits per month
after email verification, at the time of writing). The response field names
below follow that documentation; Scrappa notes its examples are
illustrative, so every field is read defensively and invalid records are
rejected rather than guessed.

Run from the command line:

    python data_fetcher.py            # respects MIN_REFRESH_INTERVAL_MINUTES
    python data_fetcher.py --force    # ignore the interval (still needs a key)

The web app never calls the API on a normal page view. It reads the CSV
written here, and only calls refresh() from the token-protected refresh
button or the optional background schedule.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config
import housing_data

SOURCE_NAME = "99acres listings via Scrappa API"
LISTINGS_FILE = "live_listings.csv"
STATUS_FILE = "fetch_status.json"
RAW_CACHE_DIR = "raw_cache"

_lock = threading.Lock()


class FetchError(Exception):
    """A failure that should be reported to the user, never a stack trace."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _paths():
    live_dir = config.LIVE_DIR
    return live_dir / LISTINGS_FILE, live_dir / STATUS_FILE, live_dir / RAW_CACHE_DIR


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------
def read_status() -> dict:
    _, status_path, _ = _paths()
    try:
        with open(status_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_status(**updates) -> dict:
    listings_path, status_path, _ = _paths()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = read_status()
    status.update(updates)
    tmp = status_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.replace(status_path)
    return status


def data_state() -> dict:
    """Describe the live data for the website: live, cached or unavailable."""
    status = read_status()
    listings = load_live_listings()
    last_success = status.get("last_success")
    state = {
        "configured": bool(config.SCRAPPA_API_KEY),
        "source": SOURCE_NAME,
        "records": len(listings),
        "last_success": last_success,
        "last_attempt": status.get("last_attempt"),
        "last_error": status.get("last_error"),
        "label": "unavailable",
    }
    if last_success and len(listings):
        age = _now() - datetime.fromisoformat(last_success)
        state["age_hours"] = round(age.total_seconds() / 3600, 1)
        state["label"] = "live" if age <= timedelta(hours=config.LIVE_FRESHNESS_HOURS) else "cached"
    return state


# ---------------------------------------------------------------------------
# Parsing and normalisation
# ---------------------------------------------------------------------------
def _area_to_sqft(area) -> float | None:
    """Area object {"unit": "sq.ft", "value": 1200} -> square feet."""
    if not isinstance(area, dict):
        return None
    value = area.get("value")
    unit = str(area.get("unit") or "sq.ft").strip().lower()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    factor = housing_data.UNIT_TO_SQFT.get(unit)
    return value * factor if factor else None


def _price_to_lakh(price) -> float | None:
    """Price object {"min": 3800000, "max": 3800000, "currency": "INR"} -> lakh."""
    if not isinstance(price, dict):
        return None
    if str(price.get("currency") or "INR").upper() != "INR":
        return None
    values = []
    for key in ("min", "max"):
        try:
            v = float(price.get(key))
            if v > 0:
                values.append(v)
        except (TypeError, ValueError):
            pass
    if not values:
        return None
    return sum(values) / len(values) / 100_000


def _clean_locality(text) -> str:
    """'Sector 14 Dwarka, Delhi' -> 'Sector 14 Dwarka'."""
    text = str(text or "").strip()
    if "," in text:
        text = text.rsplit(",", 1)[0]
    return housing_data.normalise_locality(text)


def normalise_listings(payload: dict, city_token: str, retrieved_at: str) -> tuple[pd.DataFrame, dict]:
    """Turn one API response into SCHEMA rows. Returns (valid rows, rejected counts)."""
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise FetchError("API response did not report success")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("properties"), list):
        raise FetchError("API response has no data.properties list")

    rows, rejected = [], {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for card in data["properties"]:
        if not isinstance(card, dict) or not card.get("id"):
            reject("missing id")
            continue
        if card.get("transaction_type") not in (None, "sale"):
            reject("not for sale")
            continue
        if card.get("property_category") not in (None, "residential"):
            reject("not residential")
            continue
        area = _area_to_sqft(card.get("area"))
        price = _price_to_lakh(card.get("price"))
        if area is None:
            reject("missing or unknown area unit")
            continue
        if price is None:
            reject("missing price")
            continue
        rows.append({
            "property_id": f"99acres-{card['id']}",
            "city": str(card.get("city") or data.get("city") or city_token).strip(),
            "locality": _clean_locality(card.get("locality")),
            "area_sqft": round(area, 1),
            "bedrooms": pd.to_numeric(card.get("bedrooms"), errors="coerce"),
            "bathrooms": pd.to_numeric(card.get("bathrooms"), errors="coerce"),
            "balconies": None,
            "area_type": None,
            "ready_to_move": None,
            "price_lakh": round(price, 2),
            "price_per_sqft": None,
            "source": SOURCE_NAME,
            "source_url": str(card.get("url") or ""),
            "retrieved_at": retrieved_at,
        })

    df = pd.DataFrame(rows, columns=housing_data.SCHEMA)
    if df.empty:
        return df, rejected
    df, reasons = housing_data.validate(df)
    for reason, count in reasons.items():
        rejected[reason] = rejected.get(reason, 0) + count
    return df.reset_index(drop=True), rejected


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------
def _get_json(endpoint: str, params: dict, session=None) -> dict:
    session = session or requests
    url = f"{config.SCRAPPA_BASE_URL.rstrip('/')}/{endpoint}"
    headers = {"x-api-key": config.SCRAPPA_API_KEY, "Accept": "application/json"}
    last_error = "unknown error"
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout:
            last_error = "request timed out"
        except requests.RequestException as exc:
            last_error = f"network error ({exc.__class__.__name__})"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    raise FetchError("API returned invalid JSON")
            if resp.status_code in (401, 403):
                raise FetchError("API key rejected (HTTP %d). Check SCRAPPA_API_KEY." % resp.status_code)
            if resp.status_code == 404:
                raise FetchError(f"City '{params.get('city')}' not found on 99acres (HTTP 404)")
            if resp.status_code == 422:
                raise FetchError("API rejected the request parameters (HTTP 422)")
            if resp.status_code == 402:
                raise FetchError("No API credits left (HTTP 402)")
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code} from API"
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    time.sleep(min(int(retry_after), 30))
                    continue
            else:
                raise FetchError(f"Unexpected HTTP {resp.status_code} from API")
        if attempt < config.MAX_RETRIES:
            time.sleep(2 ** attempt)  # 2s, 4s, ...
    raise FetchError(f"API unavailable after {config.MAX_RETRIES} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def load_live_listings() -> pd.DataFrame:
    """Last successfully saved live listings (empty frame if none)."""
    listings_path, _, _ = _paths()
    if not listings_path.exists():
        return pd.DataFrame(columns=housing_data.SCHEMA)
    try:
        return pd.read_csv(listings_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=housing_data.SCHEMA)


def refresh(force: bool = False, session=None) -> dict:
    """Fetch, validate, de-duplicate and save live listings.

    Never raises: returns a result dict with "ok" and a human message.
    On any failure the previous CSV is left untouched.
    """
    if not config.SCRAPPA_API_KEY:
        return {"ok": False, "message": "Live data is not configured (SCRAPPA_API_KEY is not set). "
                                         "Showing the historical dataset only."}
    if not _lock.acquire(blocking=False):
        return {"ok": False, "message": "A refresh is already running."}
    try:
        status = read_status()
        last_success = status.get("last_success")
        if last_success and not force:
            age = _now() - datetime.fromisoformat(last_success)
            if age < timedelta(minutes=config.MIN_REFRESH_INTERVAL_MINUTES):
                mins = int(age.total_seconds() // 60)
                return {"ok": True, "skipped": True,
                        "message": f"Data was refreshed {mins} minutes ago; using the cached copy "
                                   f"(minimum interval {config.MIN_REFRESH_INTERVAL_MINUTES} minutes)."}

        retrieved_at = _now().isoformat(timespec="seconds")
        _write_status(last_attempt=retrieved_at)
        listings_path, _, raw_dir = _paths()
        raw_dir.mkdir(parents=True, exist_ok=True)

        frames, rejected, errors = [], {}, []
        for city in config.LIVE_CITIES:
            try:
                payload = _get_json("city-listings", {"city": city}, session=session)
                # Keep the raw response so a problem can be inspected later.
                with open(raw_dir / f"city-listings_{city}.json", "w") as f:
                    json.dump(payload, f)
                df, rej = normalise_listings(payload, city, retrieved_at)
                frames.append(df)
                for k, v in rej.items():
                    rejected[k] = rejected.get(k, 0) + v
            except FetchError as exc:
                errors.append(f"{city}: {exc}")

        if not frames:
            message = "; ".join(errors) or "no data returned"
            _write_status(last_error=message, last_error_at=retrieved_at)
            return {"ok": False, "message": f"Refresh failed, keeping previous data. {message}"}

        new = pd.concat(frames, ignore_index=True)
        old = load_live_listings()
        old_ids = set(old["property_id"].astype(str))
        added = int((~new["property_id"].isin(old_ids)).sum())
        updated = len(new) - added
        # Newest record wins for the same listing id.
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset="property_id", keep="last")
        listings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = listings_path.with_suffix(".tmp")
        combined[housing_data.SCHEMA].to_csv(tmp, index=False)
        tmp.replace(listings_path)

        result = {"ok": True, "added": added, "updated": updated, "rejected": rejected,
                  "total": len(combined), "errors": errors}
        _write_status(last_success=retrieved_at, last_result=result,
                      last_error="; ".join(errors) or None)
        result["message"] = (f"Fetched {len(new)} valid listings ({added} new, {updated} updated, "
                             f"{sum(rejected.values())} rejected). {len(combined)} listings stored.")
        return result
    except Exception as exc:  # last line of defence: never crash the web app
        _write_status(last_error=f"unexpected error: {exc.__class__.__name__}")
        return {"ok": False, "message": "Refresh failed unexpectedly; keeping previous data."}
    finally:
        _lock.release()


def start_background_refresh(hours: float) -> threading.Thread | None:
    """Optional simple scheduler: refresh every `hours` in a daemon thread."""
    if hours <= 0 or not config.SCRAPPA_API_KEY:
        return None

    def loop():
        while True:
            refresh()
            time.sleep(hours * 3600)

    thread = threading.Thread(target=loop, name="live-data-refresh", daemon=True)
    thread.start()
    return thread


def main():
    parser = argparse.ArgumentParser(description="Refresh live listings from the Scrappa 99acres API.")
    parser.add_argument("--force", action="store_true", help="ignore the minimum refresh interval")
    args = parser.parse_args()
    result = refresh(force=args.force)
    print(result["message"])
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
