import os
import csv
import io
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import ReviewRequest, PredictionResponse, ReviewPrediction
from backend.utils import ModelPipeline, preprocess_review

pipeline = ModelPipeline(models_dir='models')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    try:
        pipeline.load_models()
    except Exception as e:
        print(f"Error loading models at startup: {e}")
        print("Starting API without loaded models. Please train models first.")
    yield

app = FastAPI(
    title="Movie Sentiment Analyzer API",
    description="Sentiment Analysis API using TF-IDF + Logistic Regression and PyTorch LSTM",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    if pipeline.lr_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run the training script first."
        )
        
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty or whitespace.")
        
    processed = preprocess_review(text)
    
    lr_sentiment, lr_conf = pipeline.predict_lr(processed)
    lstm_sentiment, lstm_conf = pipeline.predict_lstm(processed)
    
    return PredictionResponse(
        text=text,
        processed_text=processed,
        lr=ReviewPrediction(sentiment=lr_sentiment, confidence=lr_conf),
        lstm=ReviewPrediction(sentiment=lstm_sentiment, confidence=lstm_conf)
    )

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if pipeline.lr_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training script first."
        )
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    content = await file.read()
    try:
        csv_text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            csv_text = content.decode('latin-1')
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode CSV file.")
            
    reader = csv.reader(io.StringIO(csv_text))
    
    # Read headers
    try:
        headers = next(reader)
    except StopIteration:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
        
    # Determine review column index
    review_idx = -1
    for i, h in enumerate(headers):
        if h.lower() in ['review', 'text', 'body', 'content']:
            review_idx = i
            break
            
    if review_idx == -1:
        # Default to first column if no header matches
        review_idx = 0
        
    results = []
    for row in reader:
        if not row or len(row) <= review_idx:
            continue
        review_text = row[review_idx].strip()
        if len(review_text) < 3:
            continue
            
        processed = preprocess_review(review_text)
        lr_sentiment, lr_conf = pipeline.predict_lr(processed)
        lstm_sentiment, lstm_conf = pipeline.predict_lstm(processed)
        
        results.append({
            "review": review_text,
            "lr_sentiment": lr_sentiment,
            "lr_confidence": lr_conf,
            "lstm_sentiment": lstm_sentiment,
            "lstm_confidence": lstm_conf
        })
        
    return {"results": results}

@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join('models', 'metrics.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Please train models first to generate metrics."
        )
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
