"""Affordable Housing Price Estimator - Flask web application.

Start locally:   python app.py           (http://localhost:8000)
In Docker/ECS:   gunicorn (see Dockerfile)

Routes
  GET  /?city=mumbai  form for one city (and results after a POST)
  POST /              estimate a property ("city" is a form field)
  POST /api/estimate  same as above, JSON in / JSON out
  GET  /area-map      locality map image (only if GOOGLE_MAPS_API_KEY is set)
  POST /refresh       refresh live listings (needs REFRESH_TOKEN)
  GET  /health        health check for Docker and the ECS load balancer
"""
from __future__ import annotations

import hmac
import logging
from datetime import datetime

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

import apify_fetcher
import area_maps
import config
import data_fetcher
from estimator import Estimator, ModelNotReady

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("estimator-app")

app = Flask(__name__)


@app.template_filter("nicetime")
def nicetime(value):
    """'2026-10-21T03:41:53+00:00' -> '21 Oct 2026, 03:41 UTC'."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return value

# One estimator per city, loaded once per process. A city whose model is not
# trained yet maps to None and is shown as "not ready" on the website.
_estimators: dict[str, Estimator | None] = {}
_estimator_errors: dict[str, str] = {}


def get_estimator(city: str = config.DEFAULT_CITY) -> Estimator | None:
    if city not in _estimators:
        try:
            _estimators[city] = Estimator(city)
        except ModelNotReady as exc:
            _estimators[city] = None
            _estimator_errors[city] = str(exc)
            log.warning("Model not ready: %s", exc)
    return _estimators[city]


def selected_city(values) -> str:
    city = str(values.get("city") or config.DEFAULT_CITY).strip().lower()
    return city if city in config.CITIES else config.DEFAULT_CITY


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
LIMITS = {
    "area_sqft": (300, 10_000),
    "bedrooms": (1, 6),
    "bathrooms": (1, 7),
    "balconies": (0, 3),
}


def parse_form(form, est: Estimator) -> tuple[dict, list[str]]:
    """Validate only the fields this city's model uses."""
    errors, prop = [], {}
    needed = set(est.features["model"])
    locality = (form.get("locality") or "").strip()
    if not locality:
        errors.append("Please enter a locality.")
    elif len(locality) > 80:
        errors.append("Locality name is too long.")
    prop["locality"] = est.canonical_locality(locality) or locality.title()

    for field, (lo, hi) in LIMITS.items():
        if field not in needed:
            continue
        raw = (form.get(field) or "").strip()
        label = field.replace("_", " ").replace("sqft", "(sq ft)")
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"Please enter a number for {label}.")
            continue
        if not lo <= value <= hi:
            errors.append(f"{label.capitalize()} must be between {lo} and {hi}.")
        prop[field] = value if field == "area_sqft" else int(value)

    min_per_bedroom = est.spec["min_sqft_per_bedroom"]
    if "bedrooms" in prop and "area_sqft" in prop and prop["area_sqft"] / max(prop["bedrooms"], 1) < min_per_bedroom:
        errors.append(f"That area is too small for the number of bedrooms "
                      f"(minimum {min_per_bedroom} sq ft per bedroom).")

    if "area_type" in needed:
        area_type = form.get("area_type") or "Super built-up Area"
        if area_type not in ("Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"):
            errors.append("Please choose an area type from the list.")
        prop["area_type"] = area_type
    if "ready_to_move" in needed:
        prop["ready_to_move"] = 1 if form.get("ready_to_move", "1") == "1" else 0
    if "property_type" in needed:
        property_type = form.get("property_type") or (est.property_types[0] if est.property_types else "")
        if property_type not in est.property_types:
            errors.append("Please choose a property type from the list.")
        prop["property_type"] = property_type
    return prop, errors


def live_pool_for_city(city: str):
    """Validated Scrappa live listings for Bengaluru, or None.

    Mumbai needs no separate live pool: its model is trained on the
    Housing.com listings themselves, so its KNN matches are current listings.
    """
    if city != "bengaluru":
        return None
    live = data_fetcher.load_live_listings()
    if live.empty:
        return None
    first_word = live["city"].astype(str).str.lower().str.split().str[0]
    live = live[first_word.isin({"bangalore", "bengaluru"})]
    return live if len(live) else None


def page_context(city: str, **extra):
    est = get_estimator(city)
    ctx = {
        "city": city,
        "spec": config.CITIES[city],
        "cities": {key: spec["name"] for key, spec in config.CITIES.items()},
        "city_error": _estimator_errors.get(city),
        "apify_status": apify_fetcher.read_status(city) if city in apify_fetcher.ACTOR_CITY else {},
        "est": est,
        "metrics": est.metrics if est else None,
        "localities": est.localities if est else [],
        "pca_points": est.pca_points if est else None,
        "live": data_fetcher.data_state(),
        "maps_enabled": area_maps.enabled(),
        "refresh_enabled": bool(config.REFRESH_TOKEN and config.SCRAPPA_API_KEY),
        "config": config,
        "form": {},
        "errors": [],
        "result": None,
        "notice": request.args.get("notice"),
    }
    ctx.update(extra)
    return ctx


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    city = selected_city(request.values)
    est = get_estimator(city)
    if est is None:
        # No trained model for this city: explain how to set it up, and keep
        # the city selector so any city that is ready can still be used.
        return render_template("city_not_ready.html", **page_context(city)), 503
    if request.method == "GET":
        return render_template("index.html", **page_context(city))

    prop, errors = parse_form(request.form, est)
    if errors:
        return render_template("index.html", **page_context(city, form=request.form, errors=errors)), 400
    result = est.estimate(prop, live_pool=live_pool_for_city(city))
    return render_template("index.html", **page_context(city, form=request.form, result=result, prop=prop))


@app.post("/api/estimate")
def api_estimate():
    payload = request.get_json(silent=True) or {}
    city = selected_city(payload)
    est = get_estimator(city)
    if est is None:
        return jsonify({"error": f"The {config.CITIES[city]['name']} model is not available yet."}), 503
    prop, errors = parse_form({k: str(v) for k, v in payload.items()}, est)
    if errors:
        return jsonify({"errors": errors}), 400
    result = est.estimate(prop, live_pool=live_pool_for_city(city))
    result.pop("pca_point", None)
    return jsonify({"city": city, "input": prop, "estimate": result,
                    "training_data": est.metrics["dataset"].get("source"),
                    "live_data": data_fetcher.data_state() if city == "bengaluru" else None,
                    "disclaimer": "Indicative machine-learning estimate, not an official valuation."})


@app.get("/area-map")
def area_map():
    city = selected_city(request.args)
    est = get_estimator(city)
    locality = est.canonical_locality(request.args.get("locality", "")) if est else None
    if not locality:  # only localities the model knows, so the key cannot be misused
        abort(404)
    png = area_maps.get_map_png(config.CITIES[city]["name"], locality)
    if png is None:
        abort(404)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "public, max-age=604800"})


@app.post("/refresh")
def refresh():
    token = request.form.get("token", "")
    if not config.REFRESH_TOKEN or not hmac.compare_digest(token, config.REFRESH_TOKEN):
        return redirect(url_for("index", notice="Refresh not allowed: wrong or missing admin token."))
    result = data_fetcher.refresh(force=request.form.get("force") == "1")
    log.info("Manual refresh: %s", result.get("message"))
    return redirect(url_for("index", notice=result["message"]))


@app.get("/health")
def health():
    est = get_estimator(config.DEFAULT_CITY)
    cities = {city: get_estimator(city) is not None for city in config.CITIES}
    body = {"status": "ok" if est else "model_missing", "cities_ready": cities,
            "live_data": data_fetcher.data_state()["label"]}
    return jsonify(body), 200 if est else 503


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(Exception)
def handle_error(exc):
    # Log the details for the developer; show visitors a plain message only.
    if hasattr(exc, "code") and isinstance(exc.code, int) and exc.code < 500:
        return render_template("error.html", message=getattr(exc, "description", "Request error.")), exc.code
    log.exception("Unhandled error")
    return render_template("error.html", message="Something went wrong. Please try again."), 500


# Optional background refresh (AUTO_REFRESH_HOURS > 0 and an API key set).
data_fetcher.start_background_refresh(config.AUTO_REFRESH_HOURS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
