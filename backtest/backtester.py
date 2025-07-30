from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from fetchers.news_fetcher import NewsFetcher, MockNewsFetcher
from fetchers.price_fetcher import PriceFetcher, MockPriceFetcher
from analyzers.sentiment_analyzer import SentimentAnalyzer, MockSentimentAnalyzer
from engine.signal_engine import SignalEngine, SignalType, SignalResult

@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    signal: SignalType
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    metadata: Optional[Dict] = None
    # Risk management fields
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    exit_reason: str = "TIME_EXIT"  # TIME_EXIT, STOP_LOSS, TAKE_PROFIT
    hold_minutes: int = 60

@dataclass
class BacktestResult:
    trades: List[Trade]
    total_pnl: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    summary_stats: Dict
    # Enhanced statistics
    strong_signal_pnl: float = 0.0
    weak_signal_pnl: float = 0.0
    stop_loss_count: int = 0
    take_profit_count: int = 0
    avg_hold_time: float = 0.0

class Backtester:
    """Enhanced backtesting engine with risk management and dynamic entry timing"""
    
    def __init__(self, 
                 news_fetcher: NewsFetcher,
                 price_fetcher: PriceFetcher,
                 sentiment_analyzer: SentimentAnalyzer,
                 signal_engine: SignalEngine,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 # Risk management parameters
                 stop_loss_pct: float = 0.03,  # 3% stop loss
                 take_profit_pct: float = 0.05,  # 5% take profit
                 # Dynamic entry parameters
                 max_wait_minutes: int = 30,
                 quick_entry_threshold: float = 0.005,  # 0.5% for quick entry
                 quick_entry_minutes: int = 3):
        
        self.news_fetcher = news_fetcher
        self.price_fetcher = price_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_engine = signal_engine
        self.initial_capital = initial_capital
        self.position_size = position_size
        
        # Risk management
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Dynamic entry timing
        self.max_wait_minutes = max_wait_minutes
        self.quick_entry_threshold = quick_entry_threshold
        self.quick_entry_minutes = quick_entry_minutes
    
    def run(self, 
            symbols: List[str],
            start_time: datetime,
            end_time: datetime,
            interval: str = '1m',
            cached_data: Dict = None) -> BacktestResult:
        """Run enhanced backtest with risk management"""
        
        print(f"Starting enhanced backtest for {symbols} from {start_time} to {end_time}")
        
        # Fetch data
        news_data = self._fetch_news_data(symbols, start_time, end_time)
        price_data = self._fetch_price_data(symbols, interval, start_time, end_time)
        
        # Process sentiment
        sentiment_data = self._process_sentiment(news_data)
        
        # Generate signals with enhanced scoring
        signals = self._generate_signals(sentiment_data, price_data)
        
        # Execute trades with risk management
        trades = self._execute_trades_with_risk_management(signals, price_data)
        
        # Calculate performance
        result = self._calculate_enhanced_performance(trades)
        
        return result
    
    def _fetch_news_data(self, symbols: List[str], start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Fetch news data for all symbols"""
        news_data = {}
        for symbol in symbols:
            print(f"Fetching news for {symbol}...")
            try:
                news = self.news_fetcher.fetch_news(symbol, start_time, end_time)
                news_data[symbol] = news
                print(f"Found {len(news)} news articles for {symbol}")
            except Exception as e:
                print(f"Error fetching news for {symbol}: {e}")
                news_data[symbol] = []
        return news_data
    
    def _fetch_price_data(self, symbols: List[str], interval: str, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all symbols"""
        price_data = {}
        for symbol in symbols:
            print(f"Fetching price data for {symbol}...")
            try:
                prices = self.price_fetcher.fetch_prices(symbol, interval, start_time, end_time)
                price_data[symbol] = prices
                print(f"Found {len(prices)} price records for {symbol}")
            except Exception as e:
                print(f"Error fetching price data for {symbol}: {e}")
                price_data[symbol] = pd.DataFrame()
        return price_data
    
    def _process_sentiment(self, news_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Process sentiment for all news articles"""
        sentiment_data = {}
        
        for symbol, news_list in news_data.items():
            sentiment_data[symbol] = []
            
            for news in news_list:
                try:
                    if news.get('headline'):
                        sentiment_score = self.sentiment_analyzer.analyze(news['headline'])
                        
                        sentiment_data[symbol].append({
                            'timestamp': news['timestamp'],
                            'headline': news['headline'],
                            'sentiment_score': sentiment_score,
                            'confidence': 0.8  # Default confidence
                        })
                except Exception as e:
                    print(f"Error processing sentiment: {e}")
                    continue
        
        return sentiment_data
    
    def _generate_signals(self, sentiment_data: Dict[str, List[Dict]], price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """Generate trading signals with enhanced scoring"""
        signals = {}
        
        for symbol in sentiment_data.keys():
            signals[symbol] = []
            
            if symbol not in price_data:
                continue
            
            # Get all sentiment data for this symbol
            symbol_sentiments = sentiment_data[symbol]
            
            if not symbol_sentiments:
                continue
            
            # Calculate overall sentiment for the symbol
            avg_sentiment = np.mean([s['sentiment_score'] for s in symbol_sentiments])
            news_confidence = self._calculate_news_confidence(symbol_sentiments)
            
            # Get price data for the entire period
            price_df = price_data[symbol]
            
            if len(price_df) >= 10:  # Need enough price data
                # Mock market sentiment
                market_sentiment = 0.0
                
                # Generate signal using the entire price dataset
                signal = self.signal_engine.decide(
                    avg_sentiment, market_sentiment, price_df, news_confidence
                )
                
                # Use the first timestamp as signal time
                signal_time = price_df['Timestamp'].iloc[0]
                
                signals[symbol].append({
                    'timestamp': signal_time,
                    'signal': signal
                })
                
                print(f"📊 {symbol}: Sentiment={avg_sentiment:.3f}, Signal={signal.signal.value}")
        
        return signals
    
    def _calculate_news_confidence(self, window_sentiments: List[Dict]) -> float:
        """Calculate news confidence based on number of articles and sentiment consistency"""
        if not window_sentiments:
            return 0.0
        
        # Base confidence on number of articles
        num_articles = len(window_sentiments)
        base_confidence = min(1.0, num_articles / 10.0)  # Max confidence at 10+ articles
        
        # Calculate sentiment consistency (lower variance = higher confidence)
        sentiments = [s['sentiment_score'] for s in window_sentiments]
        sentiment_variance = np.var(sentiments) if len(sentiments) > 1 else 0.0
        consistency_factor = max(0.5, 1.0 - sentiment_variance)  # Higher variance reduces confidence
        
        # Calculate average confidence from individual articles
        avg_article_confidence = np.mean([s.get('confidence', 0.8) for s in window_sentiments])
        
        # Combine factors
        final_confidence = (base_confidence * 0.4 + consistency_factor * 0.3 + avg_article_confidence * 0.3)
        
        return min(1.0, final_confidence)
    
    def _group_sentiment_by_time(self, sentiment_list: List[Dict]) -> Dict[datetime, List[Dict]]:
        """Group sentiment data by time windows"""
        windows = {}
        
        for sentiment in sentiment_list:
            # Round to nearest hour
            hour = sentiment['timestamp'].replace(minute=0, second=0, microsecond=0)
            
            if hour not in windows:
                windows[hour] = []
            windows[hour].append(sentiment)
        
        return windows
    
    def _execute_trades_with_risk_management(self, signals: Dict[str, List[Dict]], price_data: Dict[str, pd.DataFrame]) -> List[Trade]:
        """Execute trades with risk management (stop loss/take profit) and dynamic entry timing"""
        trades = []
        
        # Define multiple time horizons for PnL calculation
        time_horizons = [5, 15, 30, 60]  # minutes
        
        for symbol, signal_list in signals.items():
            if symbol not in price_data:
                continue
            
            for signal_info in signal_list:
                timestamp = signal_info['timestamp']
                signal = signal_info['signal']
                
                # Skip HOLD signals
                if signal.signal == SignalType.HOLD:
                    continue
                
                # Find price at signal time
                price_at_signal = price_data[symbol][
                    price_data[symbol]['Timestamp'] >= timestamp
                ]
                
                if len(price_at_signal) > 0:
                    signal_price = price_at_signal.iloc[0]['Close']
                    
                    # Dynamic entry with risk management
                    entry_price, entry_time, exit_price, exit_time, exit_reason, hold_minutes = self._execute_trade_with_risk_management(
                        symbol, price_data[symbol], timestamp, signal.signal, signal_price, signal.confidence
                    )
                    
                    # entry/exit price가 None이면 거래 스킵
                    if entry_price is None or exit_price is None:
                        continue
                    
                    # Calculate PnL
                    if signal.signal == SignalType.LONG:
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # SHORT signals
                        pnl = (entry_price - exit_price) / entry_price
                    
                    # Create trade with risk management info
                    trade = Trade(
                        timestamp=entry_time,
                        symbol=symbol,
                        signal=signal.signal,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        confidence=signal.confidence,
                        reasoning=signal.reasoning,
                        exit_reason=exit_reason,
                        hold_minutes=hold_minutes,
                        stop_loss_price=entry_price * (1 - self.stop_loss_pct) if signal.signal == SignalType.LONG else entry_price * (1 + self.stop_loss_pct),
                        take_profit_price=entry_price * (1 + self.take_profit_pct) if signal.signal == SignalType.LONG else entry_price * (1 - self.take_profit_pct),
                        metadata={
                            'symbol_sentiment': signal.symbol_sentiment,
                            'signal_time': timestamp,
                            'entry_delay': (entry_time - timestamp).total_seconds() / 60
                        }
                    )
                    
                    trades.append(trade)
        
        return trades
    
    def _execute_trade_with_risk_management(self, symbol: str, price_df: pd.DataFrame, 
                                          signal_time: datetime, signal_type: SignalType, 
                                          signal_price: float, confidence: float) -> Tuple[Optional[float], Optional[datetime], float, datetime, str, int]:
        """
        Execute trade with dynamic entry timing and risk management
        Returns: (entry_price, entry_time, exit_price, exit_time, exit_reason, hold_minutes)
        """
        
        # Get price data after signal time
        future_prices = price_df[price_df['Timestamp'] > signal_time]
        
        if len(future_prices) == 0:
            return None, None, 0.0, signal_time, "NO_DATA", 0
        
        # Dynamic entry logic
        entry_price, entry_time = self._find_dynamic_entry(
            future_prices, signal_type, signal_price, confidence
        )
        
        if entry_price is None:
            return None, None, 0.0, signal_time, "NO_ENTRY", 0
        
        # Monitor for stop loss/take profit or time exit
        exit_price, exit_time, exit_reason, hold_minutes = self._monitor_exit_conditions(
            price_df, entry_time, entry_price, signal_type
        )
        
        return entry_price, entry_time, exit_price, exit_time, exit_reason, hold_minutes
    
    def _find_dynamic_entry(self, future_prices: pd.DataFrame, signal_type: SignalType, 
                           signal_price: float, confidence: float) -> Tuple[Optional[float], Optional[datetime]]:
        """Find entry point with dynamic timing based on price movement and confidence"""
        
        # Quick entry for high confidence signals
        if confidence > 0.7:
            max_wait = self.quick_entry_minutes
            threshold = self.quick_entry_threshold
        else:
            max_wait = self.max_wait_minutes
            threshold = self.quick_entry_threshold # Use quick_entry_threshold for all signals
        
        # Look for entry within the time limit
        for _, row in future_prices.head(max_wait).iterrows():
            current_price = row['Close']
            price_change = (current_price - signal_price) / signal_price
            
            # Check for entry based on signal type
            if signal_type == SignalType.LONG:
                # For LONG: wait for price to move up by threshold
                if price_change >= threshold:
                    return current_price, row['Timestamp']
            elif signal_type == SignalType.SHORT:
                # For SHORT: wait for price to move down by threshold
                if price_change <= -threshold:
                    return current_price, row['Timestamp']
        
        # If no entry found, use the first available price
        if len(future_prices) > 0:
            return future_prices.iloc[0]['Close'], future_prices.iloc[0]['Timestamp']
        
        return None, None
    
    def _monitor_exit_conditions(self, price_df: pd.DataFrame, entry_time: datetime, 
                               entry_price: float, signal_type: SignalType) -> Tuple[float, datetime, str, int]:
        """Monitor for stop loss, take profit, or time-based exit"""
        
        # Get prices after entry
        future_prices = price_df[price_df['Timestamp'] > entry_time]
        
        # Calculate stop loss and take profit prices
        if signal_type == SignalType.LONG:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT signals
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        # Monitor each price point
        for _, row in future_prices.iterrows():
            current_price = row['Close']
            current_time = row['Timestamp']
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            
            # Check stop loss
            if (signal_type == SignalType.LONG and current_price <= stop_loss_price) or \
               (signal_type == SignalType.SHORT and current_price >= stop_loss_price):
                return current_price, current_time, "STOP_LOSS", int(hold_minutes)
            
            # Check take profit
            if (signal_type == SignalType.LONG and current_price >= take_profit_price) or \
               (signal_type == SignalType.SHORT and current_price <= take_profit_price):
                return current_price, current_time, "TAKE_PROFIT", int(hold_minutes)
            
            # Time-based exit (60 minutes)
            if hold_minutes >= 60:
                return current_price, current_time, "TIME_EXIT", int(hold_minutes)
        
        # If no exit condition met, use the last available price
        if len(future_prices) > 0:
            last_row = future_prices.iloc[-1]
            hold_minutes = (last_row['Timestamp'] - entry_time).total_seconds() / 60
            return last_row['Close'], last_row['Timestamp'], "TIME_EXIT", int(hold_minutes)
        
        # Fallback
        return entry_price, entry_time, "NO_EXIT", 0
    
    def _calculate_enhanced_performance(self, trades: List[Trade]) -> BacktestResult:
        """Calculate enhanced performance metrics with risk management statistics"""
        if not trades:
            return BacktestResult(
                trades=[],
                total_pnl=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                summary_stats={},
                strong_signal_pnl=0.0,
                weak_signal_pnl=0.0,
                stop_loss_count=0,
                take_profit_count=0,
                avg_hold_time=0.0
            )
        
        # Calculate basic metrics
        total_pnl = sum(trade.pnl or 0.0 for trade in trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        avg_trade_pnl = total_pnl / len(trades) if trades else 0.0
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum([t.pnl or 0.0 for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl or 0.0 for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Enhanced statistics
        strong_signals = [t for t in trades if t.signal == SignalType.LONG or t.signal == SignalType.SHORT]
        weak_signals = []  # 더 이상 WEAK 신호 없음
        
        strong_signal_pnl = sum(t.pnl or 0.0 for t in strong_signals) if strong_signals else 0.0
        weak_signal_pnl = 0.0
        
        stop_loss_count = len([t for t in trades if t.exit_reason == "STOP_LOSS"])
        take_profit_count = len([t for t in trades if t.exit_reason == "TAKE_PROFIT"])
        avg_hold_time = np.mean([t.hold_minutes for t in trades]) if trades else 0.0
        
        # Summary statistics
        summary_stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t.pnl for t in trades if t.pnl and t.pnl < 0]) if any(t.pnl and t.pnl < 0 for t in trades) else 0.0,
            'profit_factor': sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0)) if any(t.pnl and t.pnl < 0 for t in trades) else float('inf'),
            'strong_signals': len(strong_signals),
            'weak_signals': 0,
            'stop_loss_rate': stop_loss_count / len(trades) if trades else 0.0,
            'take_profit_rate': take_profit_count / len(trades) if trades else 0.0
        }
        
        return BacktestResult(
            trades=trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            summary_stats=summary_stats,
            strong_signal_pnl=strong_signal_pnl,
            weak_signal_pnl=weak_signal_pnl,
            stop_loss_count=stop_loss_count,
            take_profit_count=take_profit_count,
            avg_hold_time=avg_hold_time
        ) 