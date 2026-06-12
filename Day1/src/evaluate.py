# ══════════════════════════════════════════
# FILE: src/evaluate.py
# MLOps concept: Model Evaluation & Reporting
# ══════════════════════════════════════════
import numpy as np
import json, os, pickle
import matplotlib
matplotlib.use("Agg")  # không cần GUI
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import METRICS_PATH


def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Tính đầy đủ metrics regression. Trả về dict để log MLflow."""
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100)

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def save_metrics(metrics: dict, path: str = METRICS_PATH):
    """Lưu metrics JSON — CI/CD đọc để so sánh giữa các lần train."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {path}")


def compare_with_baseline(model, X_train, X_test, y_train, y_test) -> bool:
    """
    Gate check: model PHẢI tốt hơn DummyRegressor mới được deploy.
    """
    from sklearn.dummy import DummyRegressor

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline.predict(X_test))))
    model_rmse = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    passed = model_rmse < baseline_rmse

    print(f"\n🆚 Model vs Baseline:")
    print(f"   Baseline RMSE : {baseline_rmse:.4f}")
    print(f"   Model RMSE    : {model_rmse:.4f}")
    print(f"   Improvement   : {improvement:.1f}%")
    print(f"   Gate check    : {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


def plot_predictions(model, X_test, y_test, save_path="artifacts/prediction_plot.png"):
    """Vẽ Actual vs Predicted + Residuals. Log vào MLflow như artifact."""
    preds = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, preds, alpha=0.3, s=10, color="#2d6a4f")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Price ($100k)")
    axes[0].set_ylabel("Predicted Price ($100k)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    residuals = y_test - preds
    axes[1].hist(residuals, bins=50, color="#52b788", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals Distribution")

    plt.suptitle("Model Evaluation Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved → {save_path}")
    return save_path


def plot_feature_importance(model, feature_names, save_path="artifacts/feature_importance.png"):
    """Feature importance cho tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Model không có feature_importances_")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[indices], color="#52b788")
    plt.xticks(range(len(importances)), sorted_names, rotation=30, ha="right")
    plt.title("Feature Importance")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ Feature importance saved → {save_path}")
    return save_path


def run_full_evaluation(model, X_train, X_test, y_train, y_test, feature_names, mlflow_run=None):
    """
    Full evaluation pipeline:
      1. Metrics  2. Baseline gate  3. Plots  4. Save JSON  5. Log MLflow
    """
    print("\n" + "=" * 50)
    print("🔍 RUNNING FULL EVALUATION")
    print("=" * 50)

    metrics = compute_metrics(model, X_test, y_test)
    print(f"   RMSE : {metrics['rmse']:.4f}  (≈ ±${metrics['rmse']*100_000:,.0f})")
    print(f"   MAE  : {metrics['mae']:.4f}")
    print(f"   R²   : {metrics['r2']:.4f}")
    print(f"   MAPE : {metrics['mape']:.2f}%")

    gate_passed = compare_with_baseline(model, X_train, X_test, y_train, y_test)
    metrics["gate_passed"] = gate_passed

    pred_plot = plot_predictions(model, X_test, y_test)
    fi_plot = plot_feature_importance(model, feature_names)
    save_metrics(metrics)

    if mlflow_run is not None:
        import mlflow
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifact(pred_plot)
        if fi_plot:
            mlflow.log_artifact(fi_plot)
        mlflow.log_artifact(METRICS_PATH)
        print("✅ All artifacts logged to MLflow")

    status = "✅ READY FOR STAGING" if gate_passed else "❌ DO NOT DEPLOY"
    print(f"\n📋 Evaluation complete → {status}\n" + "=" * 50)
    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import load_data, preprocess
    from src.config import MODEL_PATH

    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    if not os.path.exists(MODEL_PATH):
        print("⚠️  Chưa có model. Chạy: python main.py")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    run_full_evaluation(model, X_train, X_test, y_train, y_test, feature_names)
