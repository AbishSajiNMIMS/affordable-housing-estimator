import json

import numpy as np
import pandas as pd

from train_model import MODEL_FEATURES

PROP = {"locality": "Whitefield", "area_sqft": 1200.0, "bedrooms": 2, "bathrooms": 2, "balconies": 1,
        "area_type": "Super built-up Area", "ready_to_move": 1}


def test_artifacts_written(artifacts_dir):
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    for key in ("mae_lakh", "rmse_lakh", "r2"):
        assert key in metrics["neural_network"]["test_metrics"]
    assert len(metrics["kmeans"]["clusters"]) == 4


def test_model_loads_and_predicts_positive_price(estimator):
    price = estimator.predict_price_lakh(PROP)
    assert np.isfinite(price) and price > 0


def test_preprocessing_consistent_between_training_and_prediction(estimator):
    """The saved preprocessor must give the same matrix for the same row and
    the width the network was trained on."""
    row = pd.DataFrame([{k: PROP[k] for k in MODEL_FEATURES}])
    a = estimator.preprocessor.transform(row)
    b = estimator.preprocessor.transform(row.copy())
    assert np.array_equal(a, b)
    assert a.shape[1] == estimator.model.input_shape[-1] == estimator.metrics["neural_network"]["input_features"]


def test_unknown_locality_does_not_crash(estimator):
    price = estimator.predict_price_lakh({**PROP, "locality": "Nowhere Nagar"})
    assert price > 0


def test_similar_properties_returns_three_without_using_price(estimator):
    similar = estimator.similar_properties(PROP)
    assert len(similar) == 3
    for s in similar:
        assert s["price_lakh"] > 0 and s["source"]
    # Same inputs -> same neighbours, regardless of any price.
    assert [s["property_id"] for s in similar] == [s["property_id"] for s in estimator.similar_properties(PROP)]
    # Distances are sorted, nearest first.
    assert [s["distance"] for s in similar] == sorted(s["distance"] for s in similar)


def test_cluster_and_band(estimator):
    result = estimator.estimate(PROP)
    assert 1 <= result["cluster"]["cluster"] <= 4
    assert result["band"]["label"] in ("Lower third", "Middle third", "Upper third")
    assert len(result["pca_point"]) == 2


def test_format_inr():
    from estimator import format_inr

    assert format_inr(45.3)["rupees"] == "₹45,30,000"
    assert format_inr(250)["crore"] == "2.50 crore"
