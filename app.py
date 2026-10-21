"""Affordable Housing Price Estimator - Flask web application.

Start locally:   python app.py           (http://localhost:8000)
In Docker/ECS:   gunicorn (see Dockerfile)

Routes
  GET  /              form (and results after a POST)
  POST /              estimate a property
  POST /api/estimate  same as above, JSON in / JSON out
  POST /refresh       refresh live listings (needs REFRESH_TOKEN)
  GET  /health        health check for Docker and the ECS load balancer
"""
from __future__ import annotations

import hmac
import logging
from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, request, url_for

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

_estimator: Estimator | None = None
_estimator_error: str | None = None


def get_estimator() -> Estimator | None:
    """Load the model artifacts once per process."""
    global _estimator, _estimator_error
    if _estimator is None and _estimator_error is None:
        try:
            _estimator = Estimator()
        except ModelNotReady as exc:
            _estimator_error = str(exc)
            log.error("Model not ready: %s", exc)
    return _estimator


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
    errors, prop = [], {}
    locality = (form.get("locality") or "").strip()
    if not locality:
        errors.append("Please enter a locality.")
    elif len(locality) > 80:
        errors.append("Locality name is too long.")
    prop["locality"] = est.canonical_locality(locality) or locality.title()

    for field, (lo, hi) in LIMITS.items():
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

    if "bedrooms" in prop and "area_sqft" in prop and prop["area_sqft"] / max(prop["bedrooms"], 1) < 250:
        errors.append("That area is too small for the number of bedrooms (minimum 250 sq ft per bedroom).")

    area_type = form.get("area_type") or "Super built-up Area"
    if area_type not in ("Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"):
        errors.append("Please choose an area type from the list.")
    prop["area_type"] = area_type
    prop["ready_to_move"] = 1 if form.get("ready_to_move", "1") == "1" else 0
    return prop, errors


def live_pool_for_city():
    """Validated live listings in the training city, or None."""
    live = data_fetcher.load_live_listings()
    if live.empty:
        return None
    first_word = live["city"].astype(str).str.lower().str.split().str[0]
    live = live[first_word.isin({"bangalore", "bengaluru"})]
    return live if len(live) else None


def page_context(**extra):
    est = get_estimator()
    ctx = {
        "est": est,
        "metrics": est.metrics if est else None,
        "localities": est.localities if est else [],
        "pca_points": est.pca_points if est else None,
        "live": data_fetcher.data_state(),
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
    est = get_estimator()
    if est is None:
        return render_template("unavailable.html"), 503
    if request.method == "GET":
        return render_template("index.html", **page_context())

    prop, errors = parse_form(request.form, est)
    if errors:
        return render_template("index.html", **page_context(form=request.form, errors=errors)), 400
    result = est.estimate(prop, live_pool=live_pool_for_city())
    return render_template("index.html", **page_context(form=request.form, result=result, prop=prop))


@app.post("/api/estimate")
def api_estimate():
    est = get_estimator()
    if est is None:
        return jsonify({"error": "Model is not available yet."}), 503
    payload = request.get_json(silent=True) or {}
    prop, errors = parse_form({k: str(v) for k, v in payload.items()}, est)
    if errors:
        return jsonify({"errors": errors}), 400
    result = est.estimate(prop, live_pool=live_pool_for_city())
    result.pop("pca_point", None)
    return jsonify({"input": prop, "estimate": result, "live_data": data_fetcher.data_state(),
                    "disclaimer": "Indicative machine-learning estimate, not an official valuation."})


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
    est = get_estimator()
    body = {"status": "ok" if est else "model_missing", "live_data": data_fetcher.data_state()["label"]}
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
