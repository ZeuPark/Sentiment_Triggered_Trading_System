from typing import Any
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

@dataclass
class SignalResult:
    signal: SignalType
    confidence: float
    symbol_sentiment: float
    reasoning: str

class SignalEngine:
    def __init__(self, sentiment_threshold: float = 0.2):
        self.sentiment_threshold = sentiment_threshold

    def decide(self, symbol_sentiment: float, *args, **kwargs) -> SignalResult:
        if symbol_sentiment >= self.sentiment_threshold:
            return SignalResult(
                signal=SignalType.SHORT,
                confidence=1.0,
                symbol_sentiment=symbol_sentiment,
                reasoning=f"Sentiment {symbol_sentiment:.3f} >= {self.sentiment_threshold} → SHORT"
            )
        elif symbol_sentiment <= -self.sentiment_threshold:
            return SignalResult(
                signal=SignalType.LONG,
                confidence=1.0,
                symbol_sentiment=symbol_sentiment,
                reasoning=f"Sentiment {symbol_sentiment:.3f} <= -{self.sentiment_threshold} → LONG"
            )
        else:
            return SignalResult(
                signal=SignalType.HOLD,
                confidence=1.0,
                symbol_sentiment=symbol_sentiment,
                reasoning=f"Sentiment {symbol_sentiment:.3f} in (-{self.sentiment_threshold}, {self.sentiment_threshold}) → HOLD"
            )
    
    def generate_signal(self, sentiment_score, **kwargs):
        # Permissive logic: any nonzero sentiment triggers a trade
        if sentiment_score > 0:
            return "LONG"
        elif sentiment_score < 0:
            return "SHORT"
        else:
            return "HOLD"