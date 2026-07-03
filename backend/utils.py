import os
import re
import joblib
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn

# Ensure nltk packages are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# PyTorch LSTM Model definition (must match training script)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0.3):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)

# Constants & Preprocessing objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_review(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed)

class ModelPipeline:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.vectorizer = None
        self.lr_model = None
        self.lstm_model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        # 1. Load LR & Vectorizer
        vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.joblib')
        lr_path = os.path.join(self.models_dir, 'logistic_regression_model.joblib')
        
        if os.path.exists(vectorizer_path) and os.path.exists(lr_path):
            self.vectorizer = joblib.load(vectorizer_path)
            self.lr_model = joblib.load(lr_path)
            print("Loaded Logistic Regression & Vectorizer successfully.")
        else:
            raise FileNotFoundError("Baseline LR model or TF-IDF Vectorizer not found. Please train models first.")
            
        # 2. Load LSTM Vocab & Model
        vocab_path = os.path.join(self.models_dir, 'vocab.json')
        lstm_path = os.path.join(self.models_dir, 'lstm_model.pth')
        
        if os.path.exists(vocab_path) and os.path.exists(lstm_path):
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            
            # Recreate model structure
            self.lstm_model = SentimentLSTM(
                vocab_size=len(self.vocab),
                embedding_dim=100,
                hidden_dim=64,
                output_dim=1,
                n_layers=1
            )
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            self.lstm_model.to(self.device)
            self.lstm_model.eval()
            print("Loaded LSTM model and vocabulary successfully.")
        else:
            raise FileNotFoundError("LSTM model or Vocab not found. Please train models first.")
            
    def predict_lr(self, processed_text):
        if self.lr_model is None or self.vectorizer is None:
            raise RuntimeError("Logistic Regression model is not loaded.")
            
        features = self.vectorizer.transform([processed_text])
        prob = self.lr_model.predict_proba(features)[0][1] # Probability of positive class
        
        # Classify as positive if prob >= 0.5, else negative
        # Wait, the prompt says Positive, Negative, or Neutral.
        # How do we represent Neutral?
        # A simple way to represent Neutral is if the confidence is very close to 0.5 (e.g. between 0.45 and 0.55).
        # Let's define neutral threshold: 0.45 <= prob <= 0.55 is Neutral, > 0.55 is Positive, < 0.45 is Negative.
        # This adds an elegant Neutral classification!
        if 0.45 <= prob <= 0.55:
            sentiment = "Neutral"
            confidence = 1.0 - abs(prob - 0.5) * 2.0 # Higher confidence close to 0.5
        elif prob > 0.55:
            sentiment = "Positive"
            confidence = prob
        else:
            sentiment = "Negative"
            confidence = 1.0 - prob
            
        return sentiment, float(confidence)

    def predict_lstm(self, processed_text):
        if self.lstm_model is None or self.vocab is None:
            raise RuntimeError("LSTM model is not loaded.")
            
        # Tokenize and Pad
        seq = []
        for word in processed_text.split():
            seq.append(self.vocab.get(word, self.vocab["<UNK>"]))
            
        max_len = 150
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [self.vocab["<PAD>"]] * (max_len - len(seq))
            
        input_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            prob = self.lstm_model(input_tensor).item()
            
        if 0.45 <= prob <= 0.55:
            sentiment = "Neutral"
            confidence = 1.0 - abs(prob - 0.5) * 2.0
        elif prob > 0.55:
            sentiment = "Positive"
            confidence = prob
        else:
            sentiment = "Negative"
            confidence = 1.0 - prob
            
        return sentiment, float(confidence)
