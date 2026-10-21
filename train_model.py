"""Train every model used by the website and save the artifacts.

Run:  python train_model.py                 (Bengaluru, about a minute on a laptop)
      python train_model.py --city mumbai   (Mumbai, after apify_fetcher.py has saved listings)
      python train_model.py --all           (every city whose data is present)
      python train_model.py --include-live  (Bengaluru: also use validated Scrappa listings)

Each city gets its own models in artifacts/<city>/, trained only on that
city's data and only on the fields that city's data actually contains.

What it does, in the order taught in the FOML labs:
  1. Load and clean the labelled historical dataset (housing_data.py).
  2. Split into train/test sets (80/20).
  3. Fit the preprocessing pipeline on the training set only.
  4. Train a small Keras Sequential neural network (Lab 17 pattern) with
     MSE loss, MAE metric and EarlyStopping.
  5. Evaluate MAE, RMSE and R^2 on the held-out test set.
  6. Fit K-Means (Lab 28) and record the elbow curve (Lab 29).
  7. Fit PCA to 2 components for a visualisation (Labs 18-23).
  8. Fit LDA to classify price bands (Labs 25-26).
  9. Save everything to artifacts/ so the web app never retrains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config
import housing_data

AREA_FEATURE = ["area_sqft"]
# Bengaluru feature lists (the defaults). Other cities use the lists in config.CITIES.
NUMERIC_FEATURES = config.CITIES["bengaluru"]["numeric"]
CATEGORICAL_FEATURES = config.CITIES["bengaluru"]["categorical"]
MODEL_FEATURES = AREA_FEATURE + NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features used for K-Means and PCA (price is included on purpose here:
# these are descriptive tools, not the price predictor).
CLUSTER_FEATURES = ["log_area", "log_price"]
PCA_FEATURES = config.CITIES["bengaluru"]["pca"]
N_CLUSTERS = 4
PRICE_BANDS = ["Lower third", "Middle third", "Upper third"]
RANDOM_STATE = 42


def city_features(city: str) -> dict:
    """Feature lists for one city, saved in metrics.json for the estimator."""
    spec = config.CITIES[city]
    return {
        "numeric": spec["numeric"],
        "categorical": spec["categorical"],
        "model": AREA_FEATURE + spec["numeric"] + spec["categorical"],
        "knn": spec["knn"],
        "pca": spec["pca"],
    }


def build_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """Scaling + one-hot encoding, shared by the neural network and LDA.

    * area is log-transformed then standardised (it is very skewed),
    * other numbers are standardised (zero mean, unit variance),
    * locality and area type are one-hot encoded; localities with fewer
      than 5 training examples are grouped into one "infrequent" column,
      and unseen localities at prediction time map to that column too.
    """
    numeric = numeric or NUMERIC_FEATURES
    categorical = categorical or CATEGORICAL_FEATURES
    return ColumnTransformer(
        [
            ("area", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), AREA_FEATURE),
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
                categorical,
            ),
        ]
    )


def build_network(input_dim: int):
    """Small Sequential network, the same structure as the FOML TensorFlow lab.

    Input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear).
    The output layer is linear (no sigmoid) because we predict a number,
    not a probability. Loss is mean squared error, as noted in Lab 17
    ("regression mean_square_error").
    """
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
    return model


def regression_metrics(y_true_lakh, y_pred_lakh) -> dict:
    ape = np.abs(y_pred_lakh - y_true_lakh) / y_true_lakh
    return {
        "mae_lakh": round(float(mean_absolute_error(y_true_lakh, y_pred_lakh)), 2),
        "rmse_lakh": round(float(np.sqrt(mean_squared_error(y_true_lakh, y_pred_lakh))), 2),
        "r2": round(float(r2_score(y_true_lakh, y_pred_lakh)), 3),
        "median_abs_pct_error": round(float(np.median(ape) * 100), 1),
        "within_20_pct": round(float(np.mean(ape <= 0.20) * 100), 1),
    }


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_area"] = np.log(df["area_sqft"])
    df["log_price"] = np.log(df["price_lakh"])
    return df


def load_live_for_training() -> pd.DataFrame:
    """Validated live listings for the same city, used only with --include-live."""
    import data_fetcher

    live = data_fetcher.load_live_listings()
    if live.empty:
        return live
    aliases = {"bengaluru", "bangalore"}
    live = live[live["city"].str.lower().str.split().str[0].isin(aliases)].copy()
    # Live cards do not report balconies, area type or possession status.
    # They are filled with typical values and flagged in the source column.
    live["balconies"] = live["balconies"].fillna(live["bedrooms"].clip(upper=3) - 1)
    live["area_type"] = live["area_type"].fillna("Super built-up Area")
    live["ready_to_move"] = live["ready_to_move"].fillna(1)
    if "property_type" not in live.columns:  # files saved before this column existed
        live["property_type"] = ""
    live["source"] = live["source"] + " (asking price; some features imputed)"
    return live


def train(city: str = config.DEFAULT_CITY, epochs: int = 200, sample: int | None = None,
          include_live: bool = False, min_live_records: int = 200, artifacts_dir=None,
          data_path=None, min_rows: int | None = None, verbose: int = 1) -> dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    spec = config.CITIES[city]
    features = city_features(city)
    model_features, pca_features = features["model"], features["pca"]
    artifacts_dir = artifacts_dir or config.city_artifacts_dir(city)
    data_path = data_path or spec["data_path"]
    min_rows = config.MIN_TRAINING_ROWS if min_rows is None else min_rows
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # 1. Data -----------------------------------------------------------------
    df, report = housing_data.load_training_dataset(data_path, city=city)
    if len(df) < min_rows:
        raise ValueError(f"Only {len(df)} valid {spec['name']} listings; need at least {min_rows} to train. "
                         f"Fetch more with: {spec['how_to_get_data']}")
    os.makedirs(artifacts_dir, exist_ok=True)
    report["live_records_used"] = 0
    if include_live and city == "bengaluru":
        live = load_live_for_training()
        if len(live) < min_live_records:
            print(f"Only {len(live)} valid live records for {config.TRAINING_CITY}; "
                  f"need {min_live_records}. Training on historical data only.")
        else:
            df = pd.concat([df, live[df.columns]], ignore_index=True)
            report["live_records_used"] = len(live)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    df = add_log_columns(df)
    print(f"{spec['name']}: {len(df)} clean records  ({report})")

    # 2. Train/test split -----------------------------------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    # 3. Preprocessing (fit on training data only to avoid leakage) -----------
    preprocessor = build_preprocessor(features["numeric"], features["categorical"])
    X_train = preprocessor.fit_transform(train_df[model_features]).astype("float32")
    X_test = preprocessor.transform(test_df[model_features]).astype("float32")
    # The network learns log(price): prices span 10 lakh to 30 crore, and
    # a log scale stops the biggest properties dominating the MSE loss.
    y_train = np.log(train_df["price_lakh"].values).astype("float32")
    y_test_lakh = test_df["price_lakh"].values

    # 4. Neural network -------------------------------------------------------
    model = build_network(X_train.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32,
        callbacks=[early_stop], verbose=verbose,
    )

    # 5. Evaluation on held-out test data --------------------------------------
    nn_pred_lakh = np.exp(model.predict(X_test, verbose=0).ravel())
    nn_metrics = regression_metrics(y_test_lakh, nn_pred_lakh)

    # Simple reference points (not machine-learning models) so the network's
    # numbers can be judged: "always guess the training median" and
    # "locality median price-per-sq-ft x area".
    median_pred = np.full_like(y_test_lakh, np.median(train_df["price_lakh"]), dtype=float)
    loc_ppsf = train_df.groupby("locality")["price_per_sqft"].median()
    ppsf = test_df["locality"].map(loc_ppsf).fillna(train_df["price_per_sqft"].median())
    ppsf_pred = (ppsf * test_df["area_sqft"] / 100_000).values
    baselines = {
        "training_median_price": regression_metrics(y_test_lakh, median_pred),
        "locality_median_rate_x_area": regression_metrics(y_test_lakh, ppsf_pred),
    }
    print("Neural network test metrics:", nn_metrics)
    print("Baselines:", baselines)

    # 6. K-Means on log area and log price -------------------------------------
    cluster_scaler = StandardScaler().fit(df[CLUSTER_FEATURES])
    Xc = cluster_scaler.transform(df[CLUSTER_FEATURES])
    wcss = {}
    for k in range(1, 9):  # elbow method from Lab 29
        wcss[k] = round(float(KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xc).inertia_), 1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(Xc)
    # Renumber clusters 1..k in order of median price so the labels are readable.
    raw_labels = kmeans.labels_
    order = pd.Series(df["price_lakh"].values).groupby(raw_labels).median().sort_values().index.tolist()
    relabel = {int(old): new + 1 for new, old in enumerate(order)}
    df["cluster"] = [relabel[int(c)] for c in raw_labels]
    cluster_summary = []
    for cid, g in df.groupby("cluster"):
        cluster_summary.append({
            "cluster": int(cid),
            "count": int(len(g)),
            "median_area_sqft": int(g["area_sqft"].median()),
            "median_price_lakh": round(float(g["price_lakh"].median()), 1),
            "median_price_per_sqft": int(g["price_per_sqft"].median()),
            "typical_bedrooms": int(g["bedrooms"].median()),
            "area_range_sqft": [int(g["area_sqft"].quantile(0.1)), int(g["area_sqft"].quantile(0.9))],
            "price_range_lakh": [round(float(g["price_lakh"].quantile(0.1)), 1),
                                 round(float(g["price_lakh"].quantile(0.9)), 1)],
        })

    # 7. PCA to two dimensions ---------------------------------------------------
    pca_scaler = StandardScaler().fit(df[pca_features])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(pca_scaler.transform(df[pca_features]))
    coords = pca.transform(pca_scaler.transform(df[pca_features]))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(df), size=min(1500, len(df)), replace=False)
    pca_points = {
        "explained_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "features": pca_features,
        "loadings": {f: [round(float(pca.components_[0][i]), 3), round(float(pca.components_[1][i]), 3)]
                     for i, f in enumerate(pca_features)},
        "points": [[round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3), int(df["cluster"].iloc[i])]
                   for i in idx],
    }

    # 8. LDA: classify price band from the property characteristics -------------
    thresholds = np.quantile(train_df["price_lakh"], [1 / 3, 2 / 3])

    def band(prices):
        return np.digitize(prices, thresholds)  # 0, 1, 2

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, band(train_df["price_lakh"].values))
    lda_pred = lda.predict(X_test)
    y_band_test = band(y_test_lakh)
    lda_metrics = {
        "accuracy": round(float(accuracy_score(y_band_test, lda_pred)), 3),
        "chance_level": round(1 / 3, 3),
        "thresholds_lakh": [round(float(t), 1) for t in thresholds],
        "labels": PRICE_BANDS,
        "confusion_matrix": confusion_matrix(y_band_test, lda_pred, labels=[0, 1, 2]).tolist(),
    }
    print("LDA price-band accuracy:", lda_metrics["accuracy"])

    # 9. Save -------------------------------------------------------------------
    model.save(os.path.join(artifacts_dir, "price_nn.keras"))
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    joblib.dump({"scaler": cluster_scaler, "kmeans": kmeans, "relabel": relabel},
                os.path.join(artifacts_dir, "kmeans.joblib"))
    joblib.dump({"scaler": pca_scaler, "pca": pca}, os.path.join(artifacts_dir, "pca.joblib"))
    joblib.dump({"lda": lda, "thresholds": thresholds}, os.path.join(artifacts_dir, "lda.joblib"))
    reference_cols = housing_data.SCHEMA + ["cluster"]
    df[reference_cols].to_csv(os.path.join(artifacts_dir, "reference_data.csv"), index=False)
    with open(os.path.join(artifacts_dir, "pca_points.json"), "w") as f:
        json.dump(pca_points, f)

    with open(data_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    metrics = {
        "city": city,
        "features": features,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            **report,
            "rows_used": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "city": spec["name"],
            "source": spec["source_name"],
            "source_url": spec["source_url"],
            "license": spec["license"],
            "price_note": spec["price_note"],
            "sha256": sha,
            "sampled": bool(sample),
        },
        "neural_network": {
            "architecture": "Input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear)",
            "input_features": int(X_train.shape[1]),
            "loss": "mean_squared_error on log(price)",
            "epochs_run": len(history.history["loss"]),
            "test_metrics": nn_metrics,
        },
        "baselines": baselines,
        "kmeans": {"k": N_CLUSTERS, "features": ["log(area)", "log(price)"], "wcss_by_k": wcss,
                   "clusters": cluster_summary},
        "pca": {"explained_variance_ratio": pca_points["explained_variance_ratio"], "features": pca_features},
        "lda": lda_metrics,
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved artifacts to {artifacts_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--city", choices=sorted(config.CITIES), default=config.DEFAULT_CITY)
    parser.add_argument("--all", action="store_true",
                        help="train every city whose data is present; cities without data are skipped")
    parser.add_argument("--epochs", type=int, default=200, help="maximum epochs (early stopping usually ends sooner)")
    parser.add_argument("--sample", type=int, default=None, help="train on a random sample of N rows (for quick tests)")
    parser.add_argument("--include-live", action="store_true",
                        help="Bengaluru only: add validated Scrappa live listings to the training data")
    parser.add_argument("--min-live-records", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cities = sorted(config.CITIES) if args.all else [args.city]
    for city in cities:
        try:
            train(city=city, epochs=args.epochs, sample=args.sample, include_live=args.include_live,
                  min_live_records=args.min_live_records, verbose=0 if args.quiet else 2)
        except (FileNotFoundError, ValueError) as exc:
            if not args.all or city == config.DEFAULT_CITY:
                raise SystemExit(f"Training {city} failed: {exc}")
            # With --all a city without data is skipped; the website shows it as "not ready".
            print(f"Skipping {city}: {exc}")


if __name__ == "__main__":
    main()
