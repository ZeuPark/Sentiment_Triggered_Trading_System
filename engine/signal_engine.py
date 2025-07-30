from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    STRONG_LONG = "STRONG_LONG"
    STRONG_SHORT = "STRONG_SHORT"
    WEAK_LONG = "WEAK_LONG"
    WEAK_SHORT = "WEAK_SHORT"
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
    signal_score: float  # New field for signal strength scoring

class SignalEngine:
    """Enhanced signal generation engine with scoring system"""
    
    def __init__(self, 
                 sentiment_threshold: float = 0.3,
                 trend_lookback_periods: int = 5,
                 volume_threshold: float = 1.2,
                 news_confidence_threshold: float = 0.6,
                 price_movement_threshold: float = 0.005,
                 strong_signal_threshold: float = 0.8,  # New parameter for STRONG signals
                 price_momentum_weight: float = 0.3):   # Weight for price momentum in scoring
        """
        Initialize SignalEngine with enhanced scoring system
        
        Args:
            sentiment_threshold: Lower threshold for more sensitive signal generation (0.3)
            trend_lookback_periods: Number of periods to analyze price trend
            volume_threshold: Volume threshold for volume analysis (1.2)
            news_confidence_threshold: Minimum news confidence for signal generation (0.6)
            price_movement_threshold: Minimum price movement required for entry (0.5%)
            strong_signal_threshold: Threshold for STRONG signal classification (0.8)
            price_momentum_weight: Weight for price momentum in signal scoring (0.3)
        """
        self.sentiment_threshold = sentiment_threshold
        self.trend_lookback_periods = trend_lookback_periods
        self.volume_threshold = volume_threshold
        self.news_confidence_threshold = news_confidence_threshold
        self.price_movement_threshold = price_movement_threshold
        self.strong_signal_threshold = strong_signal_threshold
        self.price_momentum_weight = price_momentum_weight
    
    def decide(self, 
               symbol_sentiment: float, 
               market_sentiment: float, 
               price_df: pd.DataFrame,
               news_confidence: float = 0.8) -> SignalResult:
        """
        Enhanced decision logic with signal scoring system:
        - Calculate signal score = relative_sentiment * news_confidence * price_momentum
        - Score >= 0.8 → STRONG signals
        - Score >= 0.3 → WEAK signals
        - Otherwise → HOLD
        """
        
        # Calculate relative sentiment (symbol vs market)
        relative_sentiment = symbol_sentiment - market_sentiment
        
        # Analyze price trend
        price_trend = self._analyze_price_trend(price_df)
        
        # Analyze volume
        volume_signal = self._analyze_volume(price_df)
        
        # Analyze price movement for entry timing
        price_movement = self._analyze_price_movement(price_df)
        
        # Calculate price momentum (recent price change)
        price_momentum = self._calculate_price_momentum(price_df)
        
        # Enhanced decision logic with signal scoring
        signal, confidence, reasoning, signal_score = self._apply_enhanced_strategy_with_scoring(
            relative_sentiment, price_trend, volume_signal, price_movement, 
            news_confidence, price_momentum
        )
        
        return SignalResult(
            signal=signal,
            confidence=confidence,
            symbol_sentiment=symbol_sentiment,
            market_sentiment=market_sentiment,
            relative_sentiment=relative_sentiment,
            price_trend=price_trend,
            reasoning=reasoning,
            signal_score=signal_score
        )
    
    def _calculate_price_momentum(self, price_df: pd.DataFrame) -> float:
        """Calculate price momentum based on recent price changes"""
        if len(price_df) < 3:
            return 0.0
        
        # Calculate momentum as the rate of change over the last 3 periods
        recent_prices = price_df['Close'].tail(3).values
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Normalize momentum to 0-1 range
        normalized_momentum = min(1.0, abs(momentum) * 10)  # Scale by 10 for better sensitivity
        
        return normalized_momentum
    
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
    
    def _analyze_price_movement(self, price_df: pd.DataFrame) -> str:
        """Analyze price movement for entry timing"""
        if len(price_df) < 2:
            return "NEUTRAL"
        
        recent_prices = price_df['Close'].tail(2).values
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if abs(price_change) < self.price_movement_threshold:
            return "LOW_MOVEMENT"
        else:
            return "HIGH_MOVEMENT"
    
    def _apply_enhanced_strategy_with_scoring(self, 
                                            relative_sentiment: float, 
                                            price_trend: str, 
                                            volume_signal: str, 
                                            price_movement: str, 
                                            news_confidence: float,
                                            price_momentum: float) -> Tuple[SignalType, float, str, float]:
        """
        Apply enhanced strategy with signal scoring system:
        1. Calculate signal score = relative_sentiment * news_confidence * price_momentum
        2. Score >= 0.8 → STRONG signals
        3. Score >= 0.3 → WEAK signals
        4. Otherwise → HOLD
        """
        
        # Calculate signal score
        signal_score = abs(relative_sentiment) * news_confidence * (1 + price_momentum * self.price_momentum_weight)
        
        # Calculate confidence based on sentiment strength and trend alignment
        sentiment_strength = abs(relative_sentiment)
        trend_alignment = 0.0
        
        reasoning_parts = []
        
        # Determine signal type based on score and conditions
        if signal_score >= self.strong_signal_threshold:
            # STRONG signals
            if relative_sentiment >= self.sentiment_threshold and price_trend == "DOWNTREND":
                signal = SignalType.STRONG_SHORT
                trend_alignment = 1.0
                reasoning_parts.append(f"STRONG SHORT: High score ({signal_score:.2f}) with clear downtrend")
            elif relative_sentiment <= -self.sentiment_threshold and price_trend == "UPTREND":
                signal = SignalType.STRONG_LONG
                trend_alignment = 1.0
                reasoning_parts.append(f"STRONG LONG: High score ({signal_score:.2f}) with clear uptrend")
            elif relative_sentiment >= self.sentiment_threshold:
                signal = SignalType.STRONG_SHORT
                trend_alignment = 0.8
                reasoning_parts.append(f"STRONG SHORT: High score ({signal_score:.2f}) with positive sentiment")
            elif relative_sentiment <= -self.sentiment_threshold:
                signal = SignalType.STRONG_LONG
                trend_alignment = 0.8
                reasoning_parts.append(f"STRONG LONG: High score ({signal_score:.2f}) with negative sentiment")
            else:
                signal = SignalType.HOLD
                reasoning_parts.append(f"HOLD: High score but conflicting conditions")
                
        elif signal_score >= self.sentiment_threshold:
            # WEAK signals
            if relative_sentiment >= self.sentiment_threshold and price_trend in ["DOWNTREND", "NEUTRAL"]:
                signal = SignalType.WEAK_SHORT
                trend_alignment = 0.7 if price_trend == "DOWNTREND" else 0.3
                reasoning_parts.append(f"WEAK SHORT: Moderate score ({signal_score:.2f}) with {price_trend.lower()}")
            elif relative_sentiment <= -self.sentiment_threshold and price_trend in ["UPTREND", "NEUTRAL"]:
                signal = SignalType.WEAK_LONG
                trend_alignment = 0.7 if price_trend == "UPTREND" else 0.3
                reasoning_parts.append(f"WEAK LONG: Moderate score ({signal_score:.2f}) with {price_trend.lower()}")
            else:
                signal = SignalType.HOLD
                reasoning_parts.append(f"HOLD: Moderate score but no clear direction")
                
        else:
            # HOLD signals
            signal = SignalType.HOLD
            if price_movement == "LOW_MOVEMENT":
                reasoning_parts.append(f"HOLD: Low price movement")
            elif abs(relative_sentiment) < self.sentiment_threshold:
                reasoning_parts.append(f"HOLD: Weak sentiment ({relative_sentiment:.2f})")
            else:
                reasoning_parts.append(f"HOLD: Low signal score ({signal_score:.2f})")
        
        # Add volume analysis to reasoning
        if volume_signal != "NORMAL_VOLUME":
            reasoning_parts.append(f"Volume: {volume_signal}")
            if volume_signal == "HIGH_VOLUME":
                trend_alignment *= 1.2  # Boost confidence with high volume
            elif volume_signal == "LOW_VOLUME":
                trend_alignment *= 0.8  # Reduce confidence with low volume
        
        # Add news confidence to reasoning
        if news_confidence < self.news_confidence_threshold:
            reasoning_parts.append(f"Low news confidence: {news_confidence:.2f}")
            trend_alignment *= 0.7  # Reduce confidence with low news confidence
        
        # Add price momentum to reasoning
        if price_momentum > 0.5:
            reasoning_parts.append(f"High price momentum: {price_momentum:.2f}")
        
        # Calculate confidence with enhanced formula
        confidence = min(1.0, sentiment_strength * 0.4 + trend_alignment * 0.3 + 
                        (news_confidence * 0.2) + (1.0 if volume_signal == "HIGH_VOLUME" else 0.5) * 0.1)
        
        reasoning = " | ".join(reasoning_parts)
        
        return signal, confidence, reasoning, signal_score
    
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
        """Get summary statistics for signals"""
        if not results:
            return {}
        
        signal_counts = {}
        confidence_scores = []
        signal_scores = []
        
        for result in results.values():
            signal_type = result.signal.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            confidence_scores.append(result.confidence)
            signal_scores.append(result.signal_score)
        
        return {
            'signal_distribution': signal_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'avg_signal_score': np.mean(signal_scores) if signal_scores else 0.0,
            'strong_signals': signal_counts.get('STRONG_LONG', 0) + signal_counts.get('STRONG_SHORT', 0),
            'weak_signals': signal_counts.get('WEAK_LONG', 0) + signal_counts.get('WEAK_SHORT', 0),
            'total_signals': len(results)
        } 