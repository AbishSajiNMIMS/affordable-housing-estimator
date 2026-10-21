"""Locality maps from the Google Maps Static API, fetched on the server.

The browser never sees the API key: the page asks /area-map?city=..&locality=..,
this module fetches the PNG from Google once per locality, stores it in
data/live/map_cache/ and serves the cached copy afterwards. Only localities
that exist in a trained model are accepted, so the route cannot be used to
fetch arbitrary maps on your key.

The map shows the locality, not the exact property.
"""
from __future__ import annotations

import hashlib
import logging

import requests

import config

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
log = logging.getLogger("area-maps")


def enabled() -> bool:
    return bool(config.GOOGLE_MAPS_API_KEY)


def _cache_path(city_name: str, locality: str):
    digest = hashlib.sha256(f"{city_name}|{locality}".lower().encode()).hexdigest()[:24]
    return config.LIVE_DIR / "map_cache" / f"{digest}.png"


def get_map_png(city_name: str, locality: str, session=None) -> bytes | None:
    """PNG bytes for '<locality>, <city>, India', or None if unavailable."""
    if not enabled():
        return None
    path = _cache_path(city_name, locality)
    if path.exists():
        return path.read_bytes()
    params = {
        "center": f"{locality}, {city_name}, India",
        "zoom": 14,
        "size": "400x220",
        "scale": 2,
        "markers": f"color:red|{locality}, {city_name}, India",
        "key": config.GOOGLE_MAPS_API_KEY,
    }
    try:
        resp = (session or requests).get(STATIC_MAPS_URL, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        log.warning("Map request failed for %s: %s", locality, exc.__class__.__name__)
        return None
    if resp.status_code != 200 or not resp.headers.get("Content-Type", "").startswith("image/"):
        # Never log the URL: it contains the key.
        log.warning("Map not available for %s (HTTP %s)", locality, resp.status_code)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content
