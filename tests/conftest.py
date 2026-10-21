"""Shared test fixtures.

A small model is trained once per test run (3,000 rows, 5 epochs) into a
temporary folder, so the tests never touch your real artifacts/ or
data/live/ folders.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    import train_model

    path = tmp_path_factory.mktemp("artifacts")
    original = config.ARTIFACTS_DIR
    config.ARTIFACTS_DIR = path
    train_model.train(epochs=5, sample=3000, artifacts_dir=path, verbose=0)
    yield path
    config.ARTIFACTS_DIR = original


@pytest.fixture()
def live_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path / "live")
    return tmp_path / "live"


@pytest.fixture(scope="session")
def estimator(artifacts_dir):
    from estimator import Estimator

    return Estimator(artifacts_dir)


@pytest.fixture()
def client(artifacts_dir, live_dir):
    import app as app_module

    app_module._estimator = None
    app_module._estimator_error = None
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
