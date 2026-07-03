import pytest
from backend.utils import clean_text, preprocess_review, ModelPipeline

def test_clean_text():
    # Test html tag removal
    assert clean_text("<p>Hello World!</p>") == "hello world"
    # Test punctuation removal
    assert clean_text("Excellent, movie... must-watch!") == "excellent movie mustwatch"
    # Test lowercase conversion
    assert clean_text("THIS IS AWESOME") == "this is awesome"
    # Test extra whitespace removal
    assert clean_text("  so    much   space ") == "so much space"

def test_preprocess_review():
    # Test stop words removal and lemmatization
    # 'films' should become 'film', 'was' is a stop word
    text = "The films were fantastic!"
    processed = preprocess_review(text)
    assert "film" in processed
    assert "were" not in processed
    assert "fantastic" in processed

def test_empty_preprocessing():
    assert preprocess_review("") == ""
    assert preprocess_review("   ") == ""
    assert preprocess_review("<html></html>") == ""

def test_pipeline_not_loaded():
    pipeline = ModelPipeline(models_dir='invalid_dir')
    with pytest.raises(FileNotFoundError):
        pipeline.load_models()
