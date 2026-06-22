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

---

# 📓 SỔ TAY HỌC TẬP (Learning Notes)

> Ghi lại những câu hỏi & insight trong quá trình học, đặc biệt khi
> chuyển từ tư duy **software** sang tư duy **ML/MLOps**.

## 🧭 Software vs ML — những khác biệt cốt lõi

| Khía cạnh | Software thường | ML |
|-----------|----------------|-----|
| Output | Cùng input → cùng output, đúng/sai rõ ràng | Model *đoán*, có sai số; "đúng" = "đủ tốt so với baseline" |
| Bug | Thường crash hoặc sai rõ ràng | Có thể âm thầm sai (vd data leakage) — code vẫn chạy, metrics vẫn đẹp nhưng sai |
| "Build" | Biên dịch code | Train model — tốn thời gian, lặp nhiều lần, cần ghi chép |
| Artifact | File `.jar` / `.exe` | File model `.pkl` |

**Lợi thế của người có nền software:** code sạch, testing, deploy, config management.
Đây chính là phần nhiều data scientist còn yếu → là thế mạnh khi làm MLOps.

## ❓ Q&A: `mlflow ui` và `uvicorn` là gì trong ML?

Liên tưởng tới BE/FE là đúng hướng, nhưng vai trò khác:

- **`mlflow ui` (localhost:5000)** — công cụ cho *developer*, KHÔNG phải người dùng cuối.
  Như "git history nhưng cho experiments". Mỗi lần train = 1 "run" lưu params + metrics.
  Dùng để **so sánh** model nào tốt nhất. Sau khi chọn xong model thì không cần nữa khi chạy app.

- **`uvicorn src.serve:app` (localhost:8000)** — đây MỚI là Backend thật, phục vụ người dùng.
  Load model đã train, mở endpoint `/predict`. Đây là thứ chạy trên production.

- **`/docs` (Swagger UI)** — chỉ là trang để *test* API, KHÔNG phải Frontend cho người dùng.

- **Project này CHƯA có FE.** Hệ thống ML hoàn chỉnh:
  ```
  [FE: web/app]  →  [BE: serve.py - /predict]  →  [Model]
                            ↑
            [MLflow: train & chọn model — offline]
  ```

| ML | Tương đương software |
|----|---------------------|
| `mlflow ui` | CI dashboard / testing logs — chỉ dev xem |
| Quá trình train | "build & test" — offline, lặp nhiều lần |
| `best_model.pkl` | "compiled artifact" |
| `uvicorn serve.py` | Backend API thật — deploy production |
| `/docs` | API testing tool, không phải FE |

## 🔗 Q&A: Làm sao `serve.py` "nạp" được model MLflow đã train?

Mắt xích nối 2 thế giới là một **file vật lý**: `artifacts/best_model.pkl`.

**Bước 1 — `train.py` "đóng băng" model thành file** (serialize bằng pickle):
```python
with open(MODEL_PATH, "wb") as f:   # wb = write binary
    pickle.dump(best_model, f)       # biến object trong RAM → chuỗi byte trên đĩa
```
Giống compile code thành `.jar`/`.exe` — artifact tĩnh, mang đi đâu cũng chạy.

**Bước 2 — `serve.py` "rã đông" model lúc khởi động** (chỉ 1 lần):
```python
with open(MODEL_PATH, "rb") as f:   # rb = read binary
    MODEL = pickle.load(f)           # dựng lại object model trong RAM
```

**Bước 3 — mỗi request dùng model đã nạp sẵn:**
```python
input_scaled = SCALER.transform(input_array)  # scale giống lúc train
price = MODEL.predict(input_scaled)[0]
```

### ⚠️ Tại sao có 2 file `.pkl` (model + scaler)?

Lúc train, data được scale (mean=0, std=1) nên model học trên data ĐÃ scale.
→ Khi serve, input mới CŨNG phải scale y hệt, nếu không model nhận sai thang đo và đoán bậy.
`scaler.pkl` lưu đúng scaler đã fit lúc train (nhớ mean/std của tập train).

> **Quy tắc vàng:** mọi bước biến đổi data lúc train phải lặp lại y hệt lúc serve.

### ⚡ Tại sao load model 1 lần lúc startup, không load mỗi request?
```python
MODEL = pickle.load(f)   # ✅ ngoài hàm — load 1 lần, dùng cho mọi request
# ❌ load lại trong hàm predict() → chậm gấp hàng trăm lần
```
Đọc đĩa chậm; load 1 lần vào RAM rồi tái dùng → mỗi request chỉ vài ms.
(Pattern quen thuộc từ software: singleton / connection pool.)

### 🏭 Cách "xịn" hơn cho production (sẽ học ở Tuần 2)
Thay vì đọc file `.pkl` thô, serve nạp model từ **MLflow Model Registry**:
```python
import mlflow.sklearn
MODEL = mlflow.sklearn.load_model("models:/HousePrice_RandomForest/Production")
```
→ Đổi model đang chạy mà KHÔNG sửa code — chỉ cần promote model mới lên "Production" trên MLflow UI.

## 🤖 Q&A: Model tôi đang dùng là gì?

Project train **3 models** rồi tự chọn cái RMSE thấp nhất làm `best_model`:
- **Ridge** — hồi quy tuyến tính có phạt (đường thẳng). Mạnh khi data tuyến tính, đơn giản.
- **RandomForest** — nhiều cây quyết định bỏ phiếu trung bình. Mạnh với data phi tuyến.
- **GradientBoosting** — cây xây tuần tự, cây sau sửa lỗi cây trước. Thường chính xác cao nhất.

**Kiểm tra model nào đang được serve:**
```bash
python -c "import pickle; m = pickle.load(open('artifacts/best_model.pkl','rb')); print(type(m).__name__)"
```

## 📚 3 concept ML mới cần hiểu sâu (khác software)

1. **Tư duy "model là xác suất, không phải logic"** — model đoán có sai số, "đúng" = đủ tốt so với baseline → lý do tồn tại `compare_with_baseline`.
2. **Data leakage** (`data_prep.py`) — bug không báo lỗi, metrics vẫn đẹp nhưng sai. Phải split TRƯỚC, scale SAU (scaler chỉ fit trên train).
3. **4 metrics** (RMSE, MAE, R², MAPE) — hiểu trực giác: "model đoán lệch trung bình bao nhiêu, giải thích được bao nhiêu % biến động". Chưa cần học công thức toán sâu.

## 🛠️ Ghi chú cài đặt môi trường (đã gặp & xử lý)

- **Lỗi build numpy/pandas từ source** → do dùng Python 3.13 quá mới (chưa có wheel sẵn).
  Khuyến nghị dùng **Python 3.11** cho cả lộ trình (Airflow & nhiều tool MLOps chưa hỗ trợ 3.13).
- **Lỗi SSL / proxy** (`CERTIFICATE_VERIFY_FAILED`, `Cannot connect to proxy`) → mạng công ty.
  Cách xử lý:
  ```bash
  env | grep -i proxy                      # kiểm tra có proxy không
  pip install -r requirements.txt \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org  # bỏ qua verify SSL tạm thời
  export SSL_CERT_FILE=$(python -m certifi) # fix gốc trên macOS
  ```
- Sau khi cài xong, "đóng băng" môi trường: `pip freeze > requirements.lock.txt`