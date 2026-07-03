from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=10000, description="The movie review text to analyze")

class ReviewPrediction(BaseModel):
    sentiment: str = Field(..., description="Sentiment classification: Positive, Negative, or Neutral")
    confidence: float = Field(..., description="Probability score for the prediction")

class PredictionResponse(BaseModel):
    text: str
    processed_text: str
    lr: ReviewPrediction
    lstm: ReviewPrediction
