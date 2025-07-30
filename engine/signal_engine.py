from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    market_sentiment: float
    relative_sentiment: float
    price_trend: str
    reasoning: str

class SignalEngine:
    """Core signal generation engine implementing Zeu's strategy"""
    
    def __init__(self, 
                 sentiment_threshold: float = 1.0,
                 trend_lookback_periods: int = 5,
                 volume_threshold: float = 1.5):
        self.sentiment_threshold = sentiment_threshold
        self.trend_lookback_periods = trend_lookback_periods
        self.volume_threshold = volume_threshold
    
    def decide(self, 
               symbol_sentiment: float, 
               market_sentiment: float, 
               price_df: pd.DataFrame) -> SignalResult:
        """
        Core decision logic based on Zeu's strategy:
        - 상대감정 ≥ 1.0 and 하락 추세 → SHORT
        - 상대감정 ≤ -1.0 and 상승 추세 → LONG
        - Otherwise → HOLD
        """
        
        # Calculate relative sentiment (symbol vs market)
        relative_sentiment = symbol_sentiment - market_sentiment
        
        # Analyze price trend
        price_trend = self._analyze_price_trend(price_df)
        
        # Analyze volume
        volume_signal = self._analyze_volume(price_df)
        
        # Core decision logic
        signal, confidence, reasoning = self._apply_strategy(
            relative_sentiment, price_trend, volume_signal
        )
        
        return SignalResult(
            signal=signal,
            confidence=confidence,
            symbol_sentiment=symbol_sentiment,
            market_sentiment=market_sentiment,
            relative_sentiment=relative_sentiment,
            price_trend=price_trend,
            reasoning=reasoning
        )
    
    def _analyze_price_trend(self, price_df: pd.DataFrame) -> str:
        """Analyze price trend over recent periods"""
        if len(price_df) < self.trend_lookback_periods:
            return "NEUTRAL"
        
        # Get recent price data
        recent_prices = price_df['Close'].tail(self.trend_lookback_periods).values
        
        # Calculate trend indicators
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate moving averages
        short_ma = np.mean(recent_prices[-3:]) if len(recent_prices) >= 3 else recent_prices[-1]
        long_ma = np.mean(recent_prices)
        
        # Determine trend
        if price_change > 0.02 and short_ma > long_ma:  # 2% up and short MA > long MA
            return "UPTREND"
        elif price_change < -0.02 and short_ma < long_ma:  # 2% down and short MA < long MA
            return "DOWNTREND"
        else:
            return "NEUTRAL"
    
    def _analyze_volume(self, price_df: pd.DataFrame) -> str:
        """Analyze volume patterns"""
        if len(price_df) < 5:
            return "NEUTRAL"
        
        recent_volume = price_df['Volume'].tail(5).values
        avg_volume = np.mean(recent_volume)
        current_volume = recent_volume[-1]
        
        if current_volume > avg_volume * self.volume_threshold:
            return "HIGH_VOLUME"
        elif current_volume < avg_volume * 0.5:
            return "LOW_VOLUME"
        else:
            return "NORMAL_VOLUME"
    
    def _apply_strategy(self, 
                       relative_sentiment: float, 
                       price_trend: str, 
                       volume_signal: str) -> Tuple[SignalType, float, str]:
        """
        Apply enhanced Zeu's strategy with relative sentiment analysis:
        1. 상대감정 ≥ 1.0 and 하락 추세 → SHORT (강한 긍정 감정 + 하락 = 숏)
        2. 상대감정 ≤ -1.0 and 상승 추세 → LONG (강한 부정 감정 + 상승 = 롱)
        3. 상대감정 ≥ 0.5 and 중립/하락 → WEAK_SHORT
        4. 상대감정 ≤ -0.5 and 중립/상승 → WEAK_LONG
        5. Otherwise → HOLD
        """
        
        # Calculate confidence based on sentiment strength and trend alignment
        sentiment_strength = abs(relative_sentiment)
        trend_alignment = 0.0
        
        reasoning_parts = []
        
        # Strong SHORT Signal Logic (상대감정 ≥ 1.0 and 하락 추세)
        if relative_sentiment >= self.sentiment_threshold and price_trend == "DOWNTREND":
            signal = SignalType.SHORT
            trend_alignment = 1.0
            reasoning_parts.append(f"STRONG SHORT: High positive sentiment ({relative_sentiment:.2f}) with clear downtrend")
            
        # Strong LONG Signal Logic (상대감정 ≤ -1.0 and 상승 추세)
        elif relative_sentiment <= -self.sentiment_threshold and price_trend == "UPTREND":
            signal = SignalType.LONG
            trend_alignment = 1.0
            reasoning_parts.append(f"STRONG LONG: High negative sentiment ({relative_sentiment:.2f}) with clear uptrend")
            
        # Weak SHORT Signal Logic (상대감정 ≥ 0.5 and 중립/하락)
        elif relative_sentiment >= 0.5 and price_trend in ["DOWNTREND", "NEUTRAL"]:
            signal = SignalType.SHORT
            trend_alignment = 0.7 if price_trend == "DOWNTREND" else 0.3
            reasoning_parts.append(f"WEAK SHORT: Moderate positive sentiment ({relative_sentiment:.2f}) with {price_trend.lower()}")
            
        # Weak LONG Signal Logic (상대감정 ≤ -0.5 and 중립/상승)
        elif relative_sentiment <= -0.5 and price_trend in ["UPTREND", "NEUTRAL"]:
            signal = SignalType.LONG
            trend_alignment = 0.7 if price_trend == "UPTREND" else 0.3
            reasoning_parts.append(f"WEAK LONG: Moderate negative sentiment ({relative_sentiment:.2f}) with {price_trend.lower()}")
            
        # HOLD Signal Logic
        else:
            signal = SignalType.HOLD
            if abs(relative_sentiment) < 0.5:
                reasoning_parts.append(f"NEUTRAL: Weak sentiment ({relative_sentiment:.2f})")
            else:
                reasoning_parts.append(f"CONFLICT: Sentiment ({relative_sentiment:.2f}) conflicts with trend ({price_trend})")
        
        # Add volume analysis to reasoning
        if volume_signal != "NORMAL_VOLUME":
            reasoning_parts.append(f"Volume: {volume_signal}")
            if volume_signal == "HIGH_VOLUME":
                trend_alignment *= 1.2  # Boost confidence with high volume
            elif volume_signal == "LOW_VOLUME":
                trend_alignment *= 0.8  # Reduce confidence with low volume
        
        # Calculate confidence with enhanced formula
        confidence = min(1.0, sentiment_strength * 0.5 + trend_alignment * 0.4 + (1.0 if volume_signal == "HIGH_VOLUME" else 0.5) * 0.1)
        
        reasoning = " | ".join(reasoning_parts)
        
        return signal, confidence, reasoning
    
    def analyze_multiple_symbols(self, 
                               symbol_sentiments: Dict[str, float],
                               market_sentiment: float,
                               price_data: Dict[str, pd.DataFrame]) -> Dict[str, SignalResult]:
        """Analyze multiple symbols and return signals for each"""
        results = {}
        
        for symbol, sentiment in symbol_sentiments.items():
            if symbol in price_data:
                results[symbol] = self.decide(
                    sentiment, market_sentiment, price_data[symbol]
                )
        
        return results
    
    def get_signal_summary(self, results: Dict[str, SignalResult]) -> Dict:
        """Get summary of all signals"""
        summary = {
            'total_signals': len(results),
            'long_signals': 0,
            'short_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'strongest_signal': None
        }
        
        if not results:
            return summary
        
        confidences = []
        strongest_signal = None
        max_confidence = 0.0
        
        for symbol, result in results.items():
            if result.signal == SignalType.LONG:
                summary['long_signals'] += 1
            elif result.signal == SignalType.SHORT:
                summary['short_signals'] += 1
            else:
                summary['hold_signals'] += 1
            
            confidences.append(result.confidence)
            
            if result.confidence > max_confidence:
                max_confidence = result.confidence
                strongest_signal = {
                    'symbol': symbol,
                    'signal': result.signal.value,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning
                }
        
        summary['avg_confidence'] = np.mean(confidences)
        summary['strongest_signal'] = strongest_signal
        
        return summary 