"""Load the saved artifacts once and answer questions about one property.

Used by the Flask app. Nothing here trains a model: run train_model.py first.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config
from train_model import CLUSTER_FEATURES, MODEL_FEATURES, PCA_FEATURES, PRICE_BANDS

# Features used to measure "similar" properties. The price is deliberately
# NOT included: we look for homes with similar characteristics and then show
# what they were listed for.
KNN_FEATURES = ["log_area", "bedrooms", "bathrooms"]

ARTIFACT_FILES = ["price_nn.keras", "preprocessor.joblib", "kmeans.joblib", "pca.joblib",
                  "lda.joblib", "reference_data.csv", "pca_points.json", "metrics.json"]


class ModelNotReady(Exception):
    pass


def format_inr(amount_lakh: float) -> dict:
    """45.3 lakh -> {'rupees': '₹45,30,000', 'lakh': '45.3 lakh', 'crore': None}."""
    rupees = int(round(amount_lakh * 100_000, -4))
    digits = str(rupees)
    # Indian digit grouping: last 3 digits, then groups of 2.
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return {
        "rupees": f"₹{digits}",
        "lakh": f"{amount_lakh:,.1f} lakh",
        "crore": f"{amount_lakh / 100:,.2f} crore" if amount_lakh >= 100 else None,
    }


class Estimator:
    def __init__(self, artifacts_dir=None):
        self.dir = str(artifacts_dir or config.ARTIFACTS_DIR)
        missing = [f for f in ARTIFACT_FILES if not os.path.exists(os.path.join(self.dir, f))]
        if missing:
            raise ModelNotReady(f"Missing artifacts {missing}. Run: python train_model.py")

        from tensorflow import keras

        self.model = keras.models.load_model(os.path.join(self.dir, "price_nn.keras"))
        self.preprocessor = joblib.load(os.path.join(self.dir, "preprocessor.joblib"))
        self.kmeans = joblib.load(os.path.join(self.dir, "kmeans.joblib"))
        self.pca = joblib.load(os.path.join(self.dir, "pca.joblib"))
        self.lda = joblib.load(os.path.join(self.dir, "lda.joblib"))
        with open(os.path.join(self.dir, "metrics.json")) as f:
            self.metrics = json.load(f)
        with open(os.path.join(self.dir, "pca_points.json")) as f:
            self.pca_points = json.load(f)

        self.reference = pd.read_csv(os.path.join(self.dir, "reference_data.csv"))
        self.reference["log_area"] = np.log(self.reference["area_sqft"])
        self.knn_scaler = StandardScaler().fit(self.reference[KNN_FEATURES])
        self.localities = sorted(self.reference["locality"].unique().tolist())
        self._known_localities = {loc.lower(): loc for loc in self.localities}
        # Localities with enough examples to have their own one-hot column.
        encoder = self.preprocessor.named_transformers_["cat"]
        self.modelled_localities = {str(c) for c in encoder.categories_[0]} - {
            str(c) for c in (encoder.infrequent_categories_[0] if encoder.infrequent_categories_[0] is not None else [])
        }
        self.clusters = {c["cluster"]: c for c in self.metrics["kmeans"]["clusters"]}

    # ------------------------------------------------------------------
    def canonical_locality(self, text: str) -> str | None:
        return self._known_localities.get(str(text or "").strip().lower())

    def _feature_row(self, prop: dict) -> pd.DataFrame:
        return pd.DataFrame([{k: prop[k] for k in MODEL_FEATURES}])

    def predict_price_lakh(self, prop: dict) -> float:
        X = self.preprocessor.transform(self._feature_row(prop)).astype("float32")
        return float(np.exp(self.model.predict(X, verbose=0).ravel()[0]))

    def similar_properties(self, prop: dict, k: int = 3, pool: pd.DataFrame | None = None) -> list[dict]:
        """KNN: the k nearest properties by scaled area, bedrooms and bathrooms.

        Candidates are limited to the same locality when it has at least k
        records; otherwise the whole pool is searched.
        """
        pool = self.reference if pool is None else pool
        if pool is None or len(pool) == 0:
            return []
        pool = pool.dropna(subset=["area_sqft", "bedrooms", "bathrooms", "price_lakh"]).copy()
        if len(pool) == 0:
            return []
        pool["log_area"] = np.log(pool["area_sqft"])
        same = pool[pool["locality"].str.lower() == str(prop["locality"]).lower()]
        candidates = same if len(same) >= k else pool
        k = min(k, len(candidates))
        query = pd.DataFrame([{"log_area": np.log(prop["area_sqft"]), "bedrooms": prop["bedrooms"],
                               "bathrooms": prop["bathrooms"]}])
        nn = NearestNeighbors(n_neighbors=k).fit(self.knn_scaler.transform(candidates[KNN_FEATURES]))
        distances, indices = nn.kneighbors(self.knn_scaler.transform(query[KNN_FEATURES]))
        out = []
        for dist, i in zip(distances[0], indices[0]):
            row = candidates.iloc[i]
            out.append({
                "property_id": row["property_id"],
                "locality": row["locality"],
                "city": row["city"],
                "area_sqft": int(row["area_sqft"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "price_lakh": round(float(row["price_lakh"]), 1),
                "price": format_inr(float(row["price_lakh"])),
                "price_per_sqft": int(round(row["price_lakh"] * 100_000 / row["area_sqft"])),
                "source": row["source"],
                "source_url": row["source_url"] if isinstance(row["source_url"], str) else "",
                "retrieved_at": row["retrieved_at"],
                "distance": round(float(dist), 3),
                "same_locality": len(same) >= k,
            })
        return out

    def cluster_for(self, area_sqft: float, price_lakh: float) -> dict:
        X = pd.DataFrame([{"log_area": np.log(area_sqft), "log_price": np.log(price_lakh)}])[CLUSTER_FEATURES]
        raw = int(self.kmeans["kmeans"].predict(self.kmeans["scaler"].transform(X))[0])
        cid = self.kmeans["relabel"][raw]
        return self.clusters[cid]

    def pca_point(self, prop: dict, price_lakh: float) -> list[float]:
        row = pd.DataFrame([{"log_area": np.log(prop["area_sqft"]), "bedrooms": prop["bedrooms"],
                             "bathrooms": prop["bathrooms"], "balconies": prop["balconies"],
                             "log_price": np.log(price_lakh)}])[PCA_FEATURES]
        xy = self.pca["pca"].transform(self.pca["scaler"].transform(row))[0]
        return [round(float(xy[0]), 3), round(float(xy[1]), 3)]

    def price_band(self, prop: dict) -> dict:
        X = self.preprocessor.transform(self._feature_row(prop)).astype("float32")
        lda = self.lda["lda"]
        probs = lda.predict_proba(X)[0]
        band = int(np.argmax(probs))
        t = self.lda["thresholds"]
        ranges = [f"below {t[0]:.0f} lakh", f"{t[0]:.0f} to {t[1]:.0f} lakh", f"above {t[1]:.0f} lakh"]
        return {"label": PRICE_BANDS[band], "range": ranges[band], "probability": round(float(probs[band]) * 100)}

    def estimate(self, prop: dict, live_pool: pd.DataFrame | None = None) -> dict:
        price = self.predict_price_lakh(prop)
        result = {
            "price_lakh": round(price, 1),
            "price": format_inr(price),
            "price_per_sqft": int(round(price * 100_000 / prop["area_sqft"])),
            "similar": self.similar_properties(prop),
            "cluster": self.cluster_for(prop["area_sqft"], price),
            "band": self.price_band(prop),
            "pca_point": self.pca_point(prop, price),
            "locality_modelled": prop["locality"] in self.modelled_localities,
            "locality_known": self.canonical_locality(prop["locality"]) is not None,
            "live_similar": [],
        }
        if live_pool is not None and len(live_pool):
            result["live_similar"] = self.similar_properties(prop, pool=live_pool)
        return result
