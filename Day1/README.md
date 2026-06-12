# 🏠 Project 01: House Price Prediction Pipeline

ML pipeline hoàn chỉnh theo chuẩn MLOps — Ngày 1 của lộ trình DevMLOps 90 ngày.

## 📐 Kiến trúc

```
Data → Preprocess → Train (3 models) → Evaluate → Track (MLflow) → Serve (FastAPI)
```

## 🛠️ Stack

| Công cụ | Vai trò |
|---------|---------|
| scikit-learn | Train models (Ridge, RandomForest, GradientBoosting) |
| MLflow | Experiment tracking + Model registry |
| FastAPI | Serve model qua REST API |
| pytest | Unit & integration testing |

## 🚀 Quick Start

```bash
# 1. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Tạo file .env (copy từ template)
cp .env.example .env

# 4. Chạy pipeline (train + evaluate)
python main.py

# 5. Xem MLflow UI
mlflow ui                        # mở http://localhost:5000

# 6. Serve model qua API
uvicorn src.serve:app --reload   # mở http://localhost:8000/docs

# 7. Run tests
pytest tests/ -v
```

## 📁 Cấu trúc

```
project01/
├── .env                  ← config (KHÔNG commit lên git)
├── .env.example          ← template config
├── .gitignore
├── requirements.txt
├── main.py               ← entry point: chạy toàn bộ pipeline
├── src/
│   ├── config.py         ← load config từ .env
│   ├── data_prep.py      ← load + preprocess data
│   ├── train.py          ← train + MLflow tracking
│   ├── evaluate.py       ← metrics, plots, baseline gate
│   └── serve.py          ← FastAPI endpoints
├── tests/
│   └── test_pipeline.py  ← pytest suite
├── data/raw/             ← raw data (DVC track sau này)
├── artifacts/            ← model, scaler, metrics, plots
└── notebooks/            ← Jupyter explorations
```

## 🎯 5 MLOps concepts trong project này

1. **Data Pipeline** — preprocess đúng chuẩn, không data leakage (scaler chỉ fit trên train)
2. **Experiment Tracking** — MLflow log params/metrics/artifacts cho mọi run
3. **Model Registry** — version models, so sánh và chọn best
4. **Evaluation Gate** — model phải tốt hơn baseline mới được "deploy"
5. **Serving** — REST API với input validation bằng Pydantic

## 📝 Bài tập Ngày 2

- [ ] Đọc hiểu từng dòng trong `data_prep.py`
- [ ] Thêm 1 feature mới (vd: `RoomsPerPerson = AveRooms / AveOccup`)
- [ ] Chạy lại pipeline, so sánh RMSE trên MLflow UI
