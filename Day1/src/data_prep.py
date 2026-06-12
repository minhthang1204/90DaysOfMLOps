# ══════════════════════════════════════════
# FILE: src/data_prep.py
# MLOps concept: Data Pipeline
# ══════════════════════════════════════════
import pandas as pd
import numpy as np
import pickle, os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TEST_SIZE, RANDOM_STATE, SCALER_PATH


def load_data() -> pd.DataFrame:
    """Load California Housing dataset (built-in, không cần tải file)."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def preprocess(df: pd.DataFrame):
    """
    Xử lý features:
      1. Loại outliers đơn giản
      2. Split train/test (TRƯỚC khi scale — tránh data leakage)
      3. Scale features bằng StandardScaler
      4. Lưu scaler để dùng lại khi serving
    """
    # 1. Loại outliers: giá nhà bị cap ở 5.0 ($500k)
    df = df[df["MedHouseVal"] < 5.0].copy()

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # 2. Split TRƯỚC khi scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Scaler chỉ fit trên TRAIN — đây là điểm quan trọng nhất
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # chỉ transform, không fit

    # 4. Lưu scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()
