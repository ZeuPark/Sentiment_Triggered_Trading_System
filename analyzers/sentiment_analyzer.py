from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis"""
    
    @abstractmethod
    def analyze(self, text: str) -> float:
        """
        Returns sentiment score between -1.0 (strong negative) and +1.0 (strong positive)
        """
        pass
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze multiple texts and return list of sentiment scores"""
        return [self.analyze(text) for text in texts]

class FinBERTSentimentAnalyzer(SentimentAnalyzer):
    """FinBERT implementation for financial sentiment analysis"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model"""
        try:
            print(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Create pipeline with specific configuration
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading FinBERT model: {e}")
            print("Falling back to VADER sentiment analyzer")
            self.pipeline = None
    
    def analyze(self, text: str) -> float:
        """Analyze sentiment using FinBERT"""
        if self.pipeline is None:
            # Fallback to VADER
            vader = VADERSentimentAnalyzer()
            return vader.analyze(text)
        
        try:
            # Clean and truncate text
            text = self._preprocess_text(text)
            
            if len(text) > 512:
                text = text[:512]
            
            # Get prediction
            result = self.pipeline(text)[0]
            
            # FinBERT specific label mapping
            label_mapping = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Calculate weighted sentiment score
            base_score = label_mapping.get(label, 0.0)
            weighted_score = base_score * confidence
            
            # Apply confidence threshold
            if confidence < 0.3:
                weighted_score *= 0.5  # Reduce impact of low confidence predictions
            
            return weighted_score
            
        except Exception as e:
            print(f"Error in FinBERT analysis: {e}")
            return 0.0
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for FinBERT analysis"""
        import re
        
        # Remove special characters but keep important financial terms
        text = re.sub(r'[^\w\s\.\,\!\?\-\$\%]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Add financial context if missing
        if not any(word in text.lower() for word in ['stock', 'price', 'market', 'earnings', 'revenue']):
            text = f"stock market news: {text}"
        
        return text
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze multiple texts efficiently"""
        if self.pipeline is None:
            return super().analyze_batch(texts)
        
        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Batch prediction
            results = self.pipeline(processed_texts)
            
            # Process results
            scores = []
            for result in results:
                label = result['label'].lower()
                confidence = result['score']
                
                label_mapping = {
                    'positive': 1.0,
                    'negative': -1.0,
                    'neutral': 0.0
                }
                
                base_score = label_mapping.get(label, 0.0)
                weighted_score = base_score * confidence
                
                if confidence < 0.3:
                    weighted_score *= 0.5
                
                scores.append(weighted_score)
            
            return scores
            
        except Exception as e:
            print(f"Error in batch FinBERT analysis: {e}")
            return [0.0] * len(texts)

class VADERSentimentAnalyzer(SentimentAnalyzer):
    """VADER implementation for sentiment analysis"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> float:
        """Analyze sentiment using VADER"""
        try:
            scores = self.analyzer.polarity_scores(text)
            # Use compound score which ranges from -1 to +1
            return scores['compound']
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return 0.0

class MockSentimentAnalyzer(SentimentAnalyzer):
    """Mock implementation for testing"""
    
    def __init__(self, bias: float = 0.0):
        self.bias = bias
    
    def analyze(self, text: str) -> float:
        """Generate mock sentiment scores"""
        import random
        
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'success']
        negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'fail', 'crash', 'decline']
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate base sentiment
        if positive_count == 0 and negative_count == 0:
            base_sentiment = random.gauss(0, 0.1)  # Neutral with some noise
        else:
            base_sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        
        # Add bias and random noise
        sentiment = base_sentiment + self.bias + random.gauss(0, 0.1)
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, sentiment))

class SentimentAggregator:
    """Aggregates sentiment scores from multiple sources"""
    
    def __init__(self, analyzers: List[SentimentAnalyzer], weights: Optional[List[float]] = None):
        self.analyzers = analyzers
        self.weights = weights or [1.0 / len(analyzers)] * len(analyzers)
        
        if len(self.weights) != len(self.analyzers):
            raise ValueError("Number of weights must match number of analyzers")
    
    def analyze(self, text: str) -> float:
        """Aggregate sentiment scores from all analyzers"""
        scores = []
        
        for analyzer, weight in zip(self.analyzers, self.weights):
            try:
                score = analyzer.analyze(text)
                scores.append(score * weight)
            except Exception as e:
                print(f"Error in analyzer: {e}")
                continue
        
        if not scores:
            return 0.0
        
        return sum(scores)
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """Analyze multiple texts using aggregated scores"""
        return [self.analyze(text) for text in texts] 