import os
import re
import json
import urllib.request
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure nltk packages are downloaded
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Create folders if not existing
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

DATA_PATH = os.path.join('data', 'IMDB_Dataset.csv')

def standardize_df(df):
    rename_dict = {}
    for col in df.columns:
        if col.lower() in ['review', 'text', 'body', 'content']:
            rename_dict[col] = 'review'
        elif col.lower() in ['sentiment', 'label', 'class', 'target']:
            rename_dict[col] = 'sentiment'
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
        mapping = {
            '1': 'positive', '0': 'negative',
            'pos': 'positive', 'neg': 'negative',
            'positive': 'positive', 'negative': 'negative',
            '1.0': 'positive', '0.0': 'negative'
        }
        df['sentiment'] = df['sentiment'].map(mapping)
        df = df.dropna(subset=['sentiment', 'review']).reset_index(drop=True)
    return df

def download_dataset():
    if os.path.exists(DATA_PATH):
        print("IMDb dataset already exists.")
        try:
            df = pd.read_csv(DATA_PATH)
            df = standardize_df(df)
            df.to_csv(DATA_PATH, index=False)
            print(f"Verified dataset: {len(df)} records.")
            if len(df) > 0:
                return
        except Exception as e:
            print(f"Error checking existing dataset: {e}. Re-downloading...")
            try:
                os.remove(DATA_PATH)
            except:
                pass
    
    urls = [
        "https://raw.githubusercontent.com/Ankit152/IMDB-Dataset/master/IMDB%20Dataset.csv",
        "https://raw.githubusercontent.com/aws-samples/aws-sagemaker-ground-truth-recipe/master/data/IMDB%20Dataset.csv",
        "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv" # Alternate name/format
    ]
    
    print("Downloading IMDb dataset...")
    for url in urls:
        try:
            print(f"Trying to download from {url}...")
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Download successful!")
            
            df = pd.read_csv(DATA_PATH)
            df = standardize_df(df)
            df.to_csv(DATA_PATH, index=False)
            print(f"Standardized and saved dataset with {len(df)} rows.")
            if len(df) > 0:
                return
        except Exception as e:
            print(f"Failed to download or parse from {url}: {e}")
            if os.path.exists(DATA_PATH):
                try:
                    os.remove(DATA_PATH)
                except:
                    pass
                
    # Fallback: Create synthetic/small data if completely offline or download failed
    print("WARNING: All downloads failed. Creating a synthetic movie reviews dataset for training/testing.")
    synthetic_reviews = []
    sentiments = []
    positive_words = ["great", "excellent", "love", "wonderful", "amazing", "beautiful", "masterpiece", "best", "perfect", "funny"]
    negative_words = ["bad", "worst", "terrible", "boring", "awful", "waste", "horrible", "hate", "dull", "stupid"]
    
    for i in range(1000):
        is_positive = i % 2 == 0
        sentiment = "positive" if is_positive else "negative"
        words = positive_words if is_positive else negative_words
        review = f"This movie was absolutely {words[i % len(words)]}. The plot was {words[(i+1) % len(words)]} and acting was {words[(i+2) % len(words)]}."
        synthetic_reviews.append(review)
        sentiments.append(sentiment)
        
    df = pd.DataFrame({'review': synthetic_reviews, 'sentiment': sentiments})
    df.to_csv(DATA_PATH, index=False)
    print("Synthetic dataset created with 1,000 samples.")

# Text preprocessing functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_review(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    processed = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(processed)

# PyTorch LSTM Model definition
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
        # Gather the last valid output from LSTM
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)

def build_vocab(texts, max_vocab_size=10000):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:max_vocab_size - 2]:
        vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(texts, vocab, max_len=150):
    tokenized = []
    for text in texts:
        seq = []
        for word in text.split():
            seq.append(vocab.get(word, vocab["<UNK>"]))
        # Truncate
        if len(seq) > max_len:
            seq = seq[:max_len]
        # Pad
        else:
            seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
        tokenized.append(seq)
    return np.array(tokenized)

def train_lstm(X_train, y_train, vocab_size, max_len=150, epochs=3, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LSTM on {device}...")
    
    # Model params
    embedding_dim = 100
    hidden_dim = 64
    output_dim = 1
    n_layers = 1
    
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
        
    return model

def main():
    download_dataset()
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Standardize labels
    df['sentiment'] = df['sentiment'].astype(str).str.lower()
    df = df[df['sentiment'].isin(['positive', 'negative'])].reset_index(drop=True)
    
    # If the dataset is 50k, we preprocess and train Logistic Regression on full
    # For LSTM, to keep CPU training time fast, we'll train on a subset (e.g. 5,000 reviews)
    print(f"Dataset loaded: {len(df)} records.")
    
    print("Preprocessing text reviews (this might take a minute)...")
    # Preprocess all
    df['processed_review'] = df['review'].apply(preprocess_review)
    
    # Split baseline
    X_train_text, X_test_text, y_train_label, y_test_label = train_test_split(
        df['processed_review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Map labels to numeric
    y_train = (y_train_label == 'positive').astype(int).values
    y_test = (y_test_label == 'positive').astype(int).values
    
    # TF-IDF Vectorizer
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Save Vectorizer
    joblib.dump(vectorizer, os.path.join('models', 'tfidf_vectorizer.joblib'))
    
    # Train Logistic Regression
    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000, C=1.0)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Save Logistic Regression
    joblib.dump(lr_model, os.path.join('models', 'logistic_regression_model.joblib'))
    
    # Evaluate Logistic Regression
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average='binary')
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()
    
    print(f"\n--- Baseline Logistic Regression Results ---")
    print(f"Accuracy: {lr_acc:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    
    # LSTM preparation
    print("\n--- Training Deep Learning Model (LSTM) ---")
    # Build vocabulary on training set (or a subset of it to be fast)
    # We will use a subset of 5,000 for training LSTM to ensure fast execution
    lstm_subset_size = min(5000, len(X_train_text))
    X_train_lstm_text = X_train_text.iloc[:lstm_subset_size]
    y_train_lstm = y_train[:lstm_subset_size]
    
    vocab = build_vocab(X_train_lstm_text, max_vocab_size=10000)
    # Save vocabulary
    with open(os.path.join('models', 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
        
    X_train_lstm_padded = tokenize_and_pad(X_train_lstm_text, vocab, max_len=150)
    X_test_lstm_padded = tokenize_and_pad(X_test_text, vocab, max_len=150)
    
    lstm_model = train_lstm(X_train_lstm_padded, y_train_lstm, len(vocab), max_len=150, epochs=4)
    
    # Evaluate LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_lstm_padded, dtype=torch.long).to(device)
        lstm_preds_prob = lstm_model(test_inputs).cpu().numpy().squeeze()
        lstm_preds = (lstm_preds_prob >= 0.5).astype(int)
        
    lstm_acc = accuracy_score(y_test, lstm_preds)
    lstm_precision, lstm_recall, lstm_f1, _ = precision_recall_fscore_support(y_test, lstm_preds, average='binary')
    lstm_cm = confusion_matrix(y_test, lstm_preds).tolist()
    
    print(f"\n--- LSTM Results ---")
    print(f"Accuracy: {lstm_acc:.4f}")
    print(f"F1-Score: {lstm_f1:.4f}")
    
    # Save LSTM model state dict
    torch.save(lstm_model.state_dict(), os.path.join('models', 'lstm_model.pth'))
    
    # Save metrics JSON
    metrics = {
        "lr": {
            "accuracy": lr_acc,
            "precision": lr_precision,
            "recall": lr_recall,
            "f1": lr_f1,
            "confusion_matrix": lr_cm
        },
        "lstm": {
            "accuracy": lstm_acc,
            "precision": lstm_precision,
            "recall": lstm_recall,
            "f1": lstm_f1,
            "confusion_matrix": lstm_cm
        }
    }
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics and models saved successfully!")

if __name__ == "__main__":
    main()
