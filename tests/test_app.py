import data_fetcher

VALID = {"locality": "Whitefield", "area_sqft": "1200", "bedrooms": "2", "bathrooms": "2",
         "balconies": "1", "area_type": "Super built-up Area", "ready_to_move": "1"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Affordable Housing Price Estimator" in html
    assert "not an official valuation" in html
    assert "Furnishing" not in html  # not in the dataset, so not offered


def test_form_returns_estimate(client):
    r = client.post("/", data=VALID)
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "₹" in html and "lakh" in html
    assert "3 most similar recorded properties" in html
    assert "Property group" in html


def test_form_validation(client):
    r = client.post("/", data={**VALID, "area_sqft": "abc", "locality": ""})
    html = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "Please enter a locality" in html and "Please enter a number" in html


def test_api_estimate(client):
    r = client.post("/api/estimate", json=VALID)
    body = r.get_json()
    assert r.status_code == 200
    assert body["estimate"]["price_lakh"] > 0
    assert len(body["estimate"]["similar"]) == 3


def test_api_rejects_bad_input(client):
    r = client.post("/api/estimate", json={**VALID, "bedrooms": "99"})
    assert r.status_code == 400 and r.get_json()["errors"]


def test_refresh_requires_token(client, monkeypatch):
    called = []
    monkeypatch.setattr(data_fetcher, "refresh", lambda **kw: called.append(1) or {"message": "x"})
    r = client.post("/refresh", data={"token": "wrong"})
    assert r.status_code == 302 and not called


def test_unknown_page_has_no_stack_trace(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and "Traceback" not in r.get_data(as_text=True)
