 🏡 AI Real Estate Price Predictor

An end-to-end AI system that:

* Extracts structured house features from natural language (LLM)
* Predicts property price using a trained ML model
* Explains the prediction in human-readable form (LLM)
* Generates a visual representation of the house

---

## 🚀 Project Overview

This project combines:

* 🤖 **LLM (Gemini)** → Feature extraction + explanation
* 📊 **Machine Learning (Scikit-learn)** → Price prediction
* ⚡ **FastAPI** → Backend API
* 🎨 **Streamlit** → User interface
* 🐳 **Docker** → Local containerized environment
* ☁️ **Deployment** → Render (API) + Streamlit Cloud (UI)

---

## 🧠 Pipeline Architecture

### 1. Stage 1 — Feature Extraction (LLM)

User input:

```text
"3 bedroom house with 2 bathrooms, 2000 sqft, built in 2010..."
```

⬇️

LLM extracts structured data:

```json
{
  "GrLivArea": 2000,
  "BedroomAbvGr": 3,
  "FullBath": 2,
  ...
}
```

* Returns:

  * extracted fields
  * missing fields
  * assumptions

---

### 2. Stage 2 — ML Prediction

* Uses trained model (`GradientBoostingRegressor`)
* Features are preprocessed using:

  * scaling
  * encoding
  * log transforms

Outputs:

```json
{
  "predicted_price": 245000,
  "q1": 180000,
  "q3": 300000
}
```

---

### 3. Stage 3 — Explanation (LLM)

LLM explains the prediction:

* compares price to market range
* highlights key drivers
* avoids technical ML details

---

### 4. Stage 4 — Image Generation

* Gemini creates a **visual prompt**
* OpenAI (or fallback) generates the image
* Returns base64 image to UI

---

## 📁 Project Structure

```text
.
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── schemas/              # Pydantic models
│   ├── services/
│   │   ├── feature_extractor.py
│   │   ├── predictor.py
│   │   ├── interpreter.py
│   │   ├── image_generator.py
│   │   └── ...
│
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── artifacts/
│       ├── best_model.joblib
│       └── training_stats.json
│
├── ui/
│   └── app.py                # Streamlit UI
│
├── prompts/
│   ├── stage1_final.txt
│   ├── stage2_final.txt
│   └── image_prompt_v1.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone repository

```bash
git clone https://github.com/GeorgeElHajj/ai-real-estate-agent-GeorgesELHAJJ.git
cd ai-real-estate-agent-GeorgesELHAJJ
```

---

### 2. Create `.env`

```env
GEMINI_API_KEY=your_key
GOOGLE_API_KEY=your_key

GEMINI_MODEL=gemini-2.5-pro
GEMINI_STAGE1_MODEL=gemini-2.5-pro
GEMINI_STAGE2_MODEL=gemini-2.5-pro

OPENAI_API_KEY=your_openai_key
GPT_IMAGE_MODEL=gpt-image-1
```

---

## 🐳 Run with Docker (Recommended)

```bash
docker build -t estima-api .
```

Access:

* API → [http://localhost:8000/docs](http://localhost:8000/docs)
* UI → [http://localhost:8501](http://localhost:8501)

---

## 🧪 Run Without Docker

### Backend

```bash
uvicorn app.main:app --reload
```

### UI

```bash
streamlit run ui/app.py
```

---

## ☁️ Deployment

### 🔹 FastAPI (Render)

* Runtime: Python 3.11
* Build:

```bash
pip install -r requirements.txt
```

* Start:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

### 🔹 Streamlit (Streamlit Cloud)

Set environment variables:

```env
API_URL=https://ai-real-estate-agent-georgeselhajj.onrender.com/predict
COMPLETE_API_URL=https://ai-real-estate-agent-georgeselhajj.onrender.com/predict_complete
IMAGE_API_URL=https://ai-real-estate-agent-georgeselhajj.onrender.com/generate-image
```

---

## 📊 Model Details

* Dataset: Ames Housing
* Model: `GradientBoostingRegressor`
* Features:

  * OverallQual
  * GrLivArea
  * TotalBsmtSF
  * GarageCars
  * Neighborhood
  * HouseAge
  * ...

### Performance

* Validation RMSE ≈ ~23k
* Test R² ≈ ~0.91

---

## 🎯 Key Features

* ✅ Natural language → structured ML input
* ✅ Missing field detection (no silent guessing)
* ✅ Explainable predictions
* ✅ Interactive UI
* ✅ Image generation
* ✅ Fully deployable system

---

## ⚠️ Challenges Solved

* ❌ Data leakage → fixed via proper split
* ❌ Missing values → handled via pipeline
* ❌ Docker + Gemini issues → fixed API key + config
* ❌ Schema errors → removed incompatible config
* ❌ Render build failure → fixed Python version

---

## 📌 Future Improvements

* Add authentication (Keycloak)
* Improve feature engineering
* Add model monitoring
* Replace fallback image generator
* Enhance UI/UX

---

## 👤 Author

Built as part of an AI engineering project combining:

* ML pipelines
* LLM orchestration
* API design
* full-stack deployment

---

## 🏁 Final Note

This project demonstrates a **production-style AI system**, not just a model:

* data → model → API → UI → deployment

