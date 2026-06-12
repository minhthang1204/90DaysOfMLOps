# ══════════════════════════════════════════
# FILE: main.py — chạy toàn bộ pipeline
# Chạy: python main.py
# ══════════════════════════════════════════
from src.data_prep import load_data, preprocess
from src.train import train_and_track
from src.evaluate import run_full_evaluation

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 HOUSE PRICE ML PIPELINE — DAY 1")
    print("=" * 50)

    print("\n📦 Step 1: Load & preprocess data...")
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    print("\n🏋️  Step 2: Train models & track với MLflow...")
    best_model, run_id = train_and_track(X_train, X_test, y_train, y_test, feature_names)

    print("\n🔍 Step 3: Full evaluation cho best model...")
    run_full_evaluation(best_model, X_train, X_test, y_train, y_test, feature_names)

    print("\n✅ Pipeline hoàn thành!")
    print("   → Xem MLflow UI : mlflow ui  (mở http://localhost:5000)")
    print("   → Serve API     : uvicorn src.serve:app --reload")
    print("   → Run tests     : pytest tests/ -v")
