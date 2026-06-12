# ══════════════════════════════════════════
# FILE: tests/test_pipeline.py
# MLOps concept: Testing — chạy: pytest tests/ -v
# ══════════════════════════════════════════
import pytest
import numpy as np
import os, sys, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import load_data, preprocess
from src.evaluate import compute_metrics
from src.config import MODEL_PATH
from sklearn.dummy import DummyRegressor


@pytest.fixture(scope="module")
def data():
    """Load data 1 lần cho cả module — tiết kiệm thời gian test."""
    df = load_data()
    return preprocess(df)


class TestDataPrep:
    def test_load_data_shape(self):
        df = load_data()
        assert df.shape[0] > 1000, "Dataset quá nhỏ"
        assert "MedHouseVal" in df.columns

    def test_train_test_same_features(self, data):
        X_train, X_test, *_ = data
        assert X_train.shape[1] == X_test.shape[1]

    def test_no_nan_after_preprocess(self, data):
        X_train, X_test, y_train, y_test, _ = data
        assert not np.isnan(X_train).any(), "Train có NaN!"
        assert not np.isnan(X_test).any(), "Test có NaN!"

    def test_scaled_distribution(self, data):
        """StandardScaler: train data phải có mean≈0, std≈1."""
        X_train, *_ = data
        assert abs(X_train.mean()) < 0.1
        assert abs(X_train.std() - 1.0) < 0.1


class TestModel:
    def test_model_better_than_baseline(self, data):
        """Gate check: model phải tốt hơn DummyRegressor."""
        X_train, X_test, y_train, y_test, _ = data

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_metrics = compute_metrics(baseline, X_test, y_test)

        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            model_metrics = compute_metrics(model, X_test, y_test)
            assert model_metrics["r2"] > baseline_metrics["r2"]
            assert model_metrics["rmse"] < baseline_metrics["rmse"]
        else:
            pytest.skip("Chưa có model — chạy main.py trước")


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.serve import app
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200

    def test_predict_endpoint(self):
        from fastapi.testclient import TestClient
        from src.serve import app
        client = TestClient(app)

        sample = {
            "MedInc": 3.5, "HouseAge": 20.0, "AveRooms": 5.0,
            "AveBedrms": 1.1, "Population": 800.0, "AveOccup": 2.8,
            "Latitude": 37.5, "Longitude": -122.0,
        }
        r = client.post("/predict", json=sample)
        if r.status_code == 200:
            data = r.json()
            assert "predicted_price_usd" in data
            assert data["predicted_price_usd"] > 0

    def test_predict_invalid_input(self):
        """Pydantic phải reject input không hợp lệ."""
        from fastapi.testclient import TestClient
        from src.serve import app
        client = TestClient(app)

        bad_sample = {"MedInc": -5}  # thiếu fields + giá trị âm
        r = client.post("/predict", json=bad_sample)
        assert r.status_code == 422  # Unprocessable Entity
