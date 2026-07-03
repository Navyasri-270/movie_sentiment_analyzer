# SentiFilm AI — Movie Sentiment Analyzer

A premium Movie Sentiment Analysis web application that classifies reviews as **Positive**, **Negative**, or **Neutral**, providing prediction confidence levels and comparing a classical baseline ML model with a PyTorch LSTM model.

## Features

1. **Single Review Analysis**: Interactive text input to run inference and compare the TF-IDF + Logistic Regression model and PyTorch LSTM model side-by-side.
2. **Batch CSV Analysis**: Drag and drop a CSV file containing reviews to predict sentiments in bulk, with an interactive table view and downloadable CSV results.
3. **Model Performance Dashboard**: Real-time evaluation metrics (Accuracy, Precision, Recall, F1-Score) and interactive Confusion Matrices for both models.
4. **Analysis History**: A history panel displaying the last 5 analyzed reviews for easy re-inspection.

## Directory Structure

```
movie_sentiment_analyzer/
├── backend/               # FastAPI application logic
│   ├── app.py             # Main FastAPI server entrypoint
│   ├── schemas.py         # Request and response models
│   ├── utils.py           # Preprocessing utilities and prediction pipelines
│   └── tests/             # Pytest unit tests for preprocessing & endpoints
├── data/                  # Standardized movie review datasets (downloaded at training)
├── frontend/              # Web application user interface
│   ├── index.html         # Premium dark glassmorphism layout
│   ├── index.css          # Styling, hover effects, and progress animations
│   └── app.js             # API requests, LocalStorage history, and file handling
├── models/                # Saved models, TF-IDF vectorizers, vocab, and metrics
├── notebooks/             # Scripts for model experiments and EDA
│   └── train_models.py    # Robust dataset downloader and model trainer
├── README.md              # Setup and run instructions
└── requirements.txt       # Python dependencies
```

## Setup & Run Instructions

### Prerequisites
- Python 3.11.x installed.

### 1. Installation
Clone or navigate to the workspace directory and run:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models
To download the IMDb movie reviews dataset (50,000 samples) and train both models:

```bash
python notebooks/train_models.py
```
This script will:
- Download the IMDb dataset from LawrenceDuan/IMDb-Review-Analysis or local fallback.
- Standardize labels, preprocess reviews (tokenizing, lowercasing, HTML/punctuation removal, lemmatizing, and stopword removal).
- Train a **Logistic Regression** baseline model on all 50k rows.
- Train a lightweight **PyTorch LSTM** model on a CPU-friendly subset.
- Save trained weights, vocabularies, TF-IDF vectorizers, and metrics to the `/models` directory.

### 3. Run FastAPI Backend
To start the backend API:

```bash
python -m uvicorn backend.app:app --reload
```
The server will start on `http://127.0.0.1:8000`. You can inspect the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### 4. Open Frontend
Open `frontend/index.html` directly in any web browser (double-click the file or open it via a static file server) to access the sentiment analyzer application!

### 5. Running Tests
Run backend unit tests to verify preprocessing and validation endpoints:

```bash
python -m pytest
```
