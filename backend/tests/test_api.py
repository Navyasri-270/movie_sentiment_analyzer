import pytest
import json
from fastapi.testclient import TestClient
from backend.app import app, pipeline
from unittest.mock import MagicMock

client = TestClient(app)

def test_predict_validation_errors():
    # Empty review
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422 # Pydantic min_length error
    
    # Review too short
    response = client.post("/predict", json={"text": "ab"})
    assert response.status_code == 422
    
    # Review too long
    response = client.post("/predict", json={"text": "a" * 15000})
    assert response.status_code == 422

def test_predict_endpoint_success_mocked():
    # Mock pipeline prediction responses
    pipeline.lr_model = MagicMock()
    pipeline.lstm_model = MagicMock()
    pipeline.predict_lr = MagicMock(return_value=("Positive", 0.85))
    pipeline.predict_lstm = MagicMock(return_value=("Positive", 0.78))
    
    response = client.post("/predict", json={"text": "This movie was absolutely amazing!"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "This movie was absolutely amazing!"
    assert data["lr"]["sentiment"] == "Positive"
    assert abs(data["lr"]["confidence"] - 0.85) < 1e-5
    assert data["lstm"]["sentiment"] == "Positive"
    assert abs(data["lstm"]["confidence"] - 0.78) < 1e-5

def test_metrics_endpoint_success_mocked(tmp_path, monkeypatch):
    # Mocking os.path.exists and loading from file
    import os
    monkeypatch.setattr(os.path, "exists", lambda x: True)
    
    mock_metrics = {
        "lr": {"accuracy": 0.89, "precision": 0.88, "recall": 0.90, "f1": 0.89, "confusion_matrix": [[4000, 1000], [500, 4500]]},
        "lstm": {"accuracy": 0.86, "precision": 0.85, "recall": 0.87, "f1": 0.86, "confusion_matrix": [[3900, 1100], [700, 4300]]}
    }
    
    # Mock the JSON loading
    import builtins
    original_open = builtins.open
    
    def mocked_open(file, mode='r', *args, **kwargs):
        if "metrics.json" in str(file):
            import io
            return io.StringIO(json.dumps(mock_metrics))
        return original_open(file, mode, *args, **kwargs)
        
    monkeypatch.setattr(builtins, "open", mocked_open)
    
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["lr"]["accuracy"] == 0.89
