# 🗺️ LỘ TRÌNH HỌC AI CODE + DevMLOps
# Dành cho: Trung cấp Python/OOP | 2–3h/ngày | 90 ngày

---

## 📌 TỔNG QUAN 3 GIAI ĐOẠN

| Giai đoạn | Thời gian | Mục tiêu |
|-----------|-----------|----------|
| Phase 1: ML Foundations + Tooling | Ngày 1–30  | Build & track model đúng chuẩn |
| Phase 2: MLOps Core               | Ngày 31–60 | Pipeline tự động, CI/CD, containerize |
| Phase 3: Production & Scale       | Ngày 61–90 | Deploy cloud, monitor, LLMOps |

---

## 🟢 PHASE 1: ML FOUNDATIONS + TOOLING (Ngày 1–30)

### Tuần 1 (Ngày 1–7): ML Pipeline chuẩn
- **Ngày 1–2:** Setup môi trường (venv, git, VSCode). Chạy Project 01 (House Price)
- **Ngày 3–4:** Hiểu data prep: missing values, encoding, scaling, train/test split
- **Ngày 5–6:** Train nhiều model, so sánh RMSE/R². Viết evaluate.py
- **Ngày 7:**   Refactor code theo OOP — tạo class DataPipeline, class Trainer

### Tuần 2 (Ngày 8–14): Experiment Tracking với MLflow
- **Ngày 8–9:**   Cài MLflow, log params/metrics/artifacts
- **Ngày 10–11:** MLflow UI — compare runs, visualize metrics
- **Ngày 12–13:** MLflow Model Registry — register, stage (Staging → Production)
- **Ngày 14:**    Mini-project: thêm hyperparameter tuning (GridSearchCV) + log vào MLflow

### Tuần 3 (Ngày 15–21): Testing & Code Quality
- **Ngày 15–16:** pytest cơ bản — unit test cho data_prep, model
- **Ngày 17–18:** Test coverage, fixtures, parametrize
- **Ngày 19–20:** Code quality: ruff/flake8, black formatter, pre-commit hooks
- **Ngày 21:**    Viết đầy đủ test suite cho Project 01 (>80% coverage)

### Tuần 4 (Ngày 22–30): Serving với FastAPI
- **Ngày 22–23:** FastAPI cơ bản — routes, Pydantic schemas, async
- **Ngày 24–25:** Load model + serve predictions, Swagger UI (/docs)
- **Ngày 26–27:** Input validation, error handling, logging
- **Ngày 28–29:** Viết API tests với TestClient
- **Ngày 30:**    🎯 **Checkpoint 1**: Pipeline hoàn chỉnh End-to-End + API chạy được

---

## 🟡 PHASE 2: MLOps CORE (Ngày 31–60)

### Tuần 5 (Ngày 31–37): Docker & Containerization
- **Ngày 31–32:** Docker cơ bản — image, container, Dockerfile
- **Ngày 33–34:** Containerize ML model + FastAPI app
- **Ngày 35–36:** Docker Compose — chạy app + MLflow server cùng nhau
- **Ngày 37:**    Push image lên Docker Hub

### Tuần 6 (Ngày 38–44): CI/CD với GitHub Actions
- **Ngày 38–39:** Git workflow — branches, PRs, merge strategy
- **Ngày 40–41:** GitHub Actions: chạy pytest tự động khi push
- **Ngày 42–43:** CI pipeline: lint → test → build Docker image
- **Ngày 44:**    CD pipeline: auto-deploy khi merge vào main

### Tuần 7 (Ngày 45–51): Data Versioning & Feature Store
- **Ngày 45–46:** DVC (Data Version Control) — track data như git track code
- **Ngày 47–48:** DVC pipelines — định nghĩa stages, dependencies
- **Ngày 49–50:** Feature Store concept — tại sao cần, Feast cơ bản
- **Ngày 51:**    Tích hợp DVC vào Project 01

### Tuần 8 (Ngày 52–60): Orchestration với Airflow
- **Ngày 52–53:** Apache Airflow — DAGs, operators, scheduling
- **Ngày 54–55:** Viết DAG cho ML pipeline: data → train → evaluate → register
- **Ngày 56–57:** Airflow sensors, branching, retry logic
- **Ngày 58–59:** Trigger retraining khi data mới về
- **Ngày 60:**    🎯 **Checkpoint 2**: Pipeline tự động chạy mỗi ngày qua Airflow

---

## 🔴 PHASE 3: PRODUCTION & SCALE (Ngày 61–90)

### Tuần 9 (Ngày 61–67): Cloud Deployment
- **Ngày 61–62:** AWS/GCP cơ bản — IAM, S3/GCS, EC2/VM
- **Ngày 63–64:** Deploy FastAPI lên cloud (EC2 hoặc Cloud Run)
- **Ngày 65–66:** MLflow tracking server trên cloud
- **Ngày 67:**    Setup domain + HTTPS với nginx

### Tuần 10 (Ngày 68–74): Model Monitoring
- **Ngày 68–69:** Data drift detection — EvidentlyAI cơ bản
- **Ngày 70–71:** Model performance monitoring — track RMSE theo thời gian
- **Ngày 72–73:** Alert khi model degradation — gửi Slack/email
- **Ngày 74:**    Dashboard monitoring với Grafana hoặc Evidently

### Tuần 11 (Ngày 75–81): LLMOps (AI Code)
- **Ngày 75–76:** LangChain/LlamaIndex cơ bản — chains, prompts, memory
- **Ngày 77–78:** RAG pipeline — embed documents, vector store (FAISS/ChromaDB)
- **Ngày 79–80:** LLM evaluation — ragas, trulens
- **Ngày 81:**    Deploy RAG app đơn giản

### Tuần 12 (Ngày 82–90): Capstone Project
- **Ngày 82–84:** Thiết kế end-to-end ML system (tự chọn bài toán)
- **Ngày 85–87:** Implement: data pipeline → train → serve → monitor
- **Ngày 88–89:** CI/CD + Docker + deploy cloud
- **Ngày 90:**    🎯 **Final**: Demo project + viết README + push GitHub

---

## 📦 DANH SÁCH 4 PROJECT THEO LỘ TRÌNH

| # | Project | Phase | Concepts |
|---|---------|-------|----------|
| 01 | House Price Prediction Pipeline | 1 | sklearn, MLflow, FastAPI, pytest |
| 02 | Churn Prediction + Auto Retraining | 2 | Docker, CI/CD, Airflow, DVC |
| 03 | Real-time Sentiment API + Monitoring | 3 | Cloud deploy, drift detection, Grafana |
| 04 | RAG Chatbot cho tài liệu nội bộ | 3 | LangChain, FAISS, LLMOps, evaluation |

---

## 🛠️ TECH STACK THEO TỪNG PHASE

```
Phase 1:  Python · scikit-learn · MLflow · FastAPI · pytest · pandas · numpy
Phase 2:  Docker · GitHub Actions · Apache Airflow · DVC · Feast
Phase 3:  AWS/GCP · EvidentlyAI · Grafana · LangChain · ChromaDB · ragas
```

---

## 📚 TÀI LIỆU HỌC CHÍNH

- **MLflow docs:** https://mlflow.org/docs/latest/index.html
- **FastAPI docs:** https://fastapi.tiangolo.com
- **Airflow docs:** https://airflow.apache.org/docs/
- **DVC docs:** https://dvc.org/doc
- **Made With ML (MLOps course):** https://madewithml.com
- **Full Stack Deep Learning:** https://fullstackdeeplearning.com

---

## ✅ NGUYÊN TẮC HỌC HIỆU QUẢ

1. **Code mỗi ngày** — dù chỉ 30 phút, không bỏ ngày nào
2. **Push GitHub mỗi ngày** — portfolio tự build theo thời gian
3. **Ghi README** — giải thích project bằng tiếng Anh, tốt cho portfolio
4. **Không copy-paste** — gõ tay để nhớ syntax và hiểu flow
5. **Sau mỗi checkpoint** — review lại, refactor code cũ trước khi đi tiếp