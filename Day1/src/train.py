# ══════════════════════════════════════════
# FILE: src/train.py
# MLOps concepts: Experiment Tracking + Model Registry
# ══════════════════════════════════════════
import mlflow
import mlflow.sklearn
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.config import MLFLOW_EXPERIMENT_NAME, MODEL_PATH, N_ESTIMATORS, RANDOM_STATE
from src.evaluate import compute_metrics


def train_and_track(X_train, X_test, y_train, y_test, feature_names):
    """
    Train nhiều models, log tất cả vào MLflow, chọn best theo RMSE.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    experiments = [
        ("Ridge", Ridge(alpha=1.0)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)),
    ]

    best_model, best_rmse, best_run_id = None, float("inf"), None

    for name, model in experiments:
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = compute_metrics(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"HousePrice_{name}",
            )

            print(f"  [{name}] RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n🏆 Best model: RMSE={best_rmse:.4f} | run_id={best_run_id}")
    return best_model, best_run_id
