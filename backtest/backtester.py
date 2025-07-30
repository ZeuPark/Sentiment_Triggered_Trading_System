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

class Backtester:
    """Backtesting engine for sentiment-based trading strategy"""
    
    def __init__(self, 
                 news_fetcher: NewsFetcher,
                 price_fetcher: PriceFetcher,
                 sentiment_analyzer: SentimentAnalyzer,
                 signal_engine: SignalEngine,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1):
        
        self.news_fetcher = news_fetcher
        self.price_fetcher = price_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_engine = signal_engine
        self.initial_capital = initial_capital
        self.position_size = position_size
    
    def run(self, 
            symbols: List[str],
            start_time: datetime,
            end_time: datetime,
            interval: str = '1m') -> BacktestResult:
        """
        Run backtest simulation
        
        Args:
            symbols: List of symbols to trade
            start_time: Start of backtest period
            end_time: End of backtest period
            interval: Price data interval
            
        Returns:
            BacktestResult with all trade data and performance metrics
        """
        
        print(f"Starting backtest for {symbols} from {start_time} to {end_time}")
        
        # Fetch all data
        news_data = self._fetch_news_data(symbols, start_time, end_time)
        price_data = self._fetch_price_data(symbols, interval, start_time, end_time)
        
        # Process news and generate signals
        sentiment_data = self._process_sentiment(news_data)
        
        # Generate trading signals
        signals = self._generate_signals(sentiment_data, price_data)
        
        # Execute trades
        trades = self._execute_trades(signals, price_data)
        
        # Calculate performance metrics
        result = self._calculate_performance(trades)
        
        # Store additional data for visualization
        result.sentiment_data = sentiment_data
        result.symbol_signals = signals
        result.price_data = price_data
        
        return result
    
    def _fetch_news_data(self, symbols: List[str], start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
        """Fetch news data for all symbols"""
        news_data = {}
        
        for symbol in symbols:
            print(f"Fetching news for {symbol}...")
            news_data[symbol] = self.news_fetcher.fetch_news(symbol, start_time, end_time)
            print(f"Found {len(news_data[symbol])} news articles for {symbol}")
        
        return news_data
    
    def _fetch_price_data(self, symbols: List[str], interval: str, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all symbols"""
        price_data = {}
        
        for symbol in symbols:
            print(f"Fetching price data for {symbol}...")
            price_data[symbol] = self.price_fetcher.fetch_prices(symbol, interval, start_time, end_time)
            print(f"Found {len(price_data[symbol])} price records for {symbol}")
        
        return price_data
    
    def _process_sentiment(self, news_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Process news data and add sentiment scores"""
        sentiment_data = {}
        
        for symbol, news_list in news_data.items():
            sentiment_data[symbol] = []
            
            for news in news_list:
                # Analyze sentiment for headline and content
                headline_sentiment = self.sentiment_analyzer.analyze(news['headline'])
                content_sentiment = self.sentiment_analyzer.analyze(news.get('content', ''))
                
                # Weighted average (headline more important)
                weighted_sentiment = headline_sentiment * 0.7 + content_sentiment * 0.3
                
                sentiment_data[symbol].append({
                    **news,
                    'sentiment_score': weighted_sentiment
                })
        
        return sentiment_data
    
    def _generate_signals(self, sentiment_data: Dict[str, List[Dict]], price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[SignalResult]]:
        """Generate trading signals based on sentiment and price data"""
        signals = {}
        
        for symbol in sentiment_data.keys():
            signals[symbol] = []
            
            if symbol not in price_data:
                continue
            
            # Group sentiment by time windows (e.g., hourly)
            sentiment_windows = self._group_sentiment_by_time(sentiment_data[symbol])
            
            for window_start, window_sentiments in sentiment_windows.items():
                # Calculate average sentiment for the window
                if window_sentiments:
                    avg_sentiment = np.mean([s['sentiment_score'] for s in window_sentiments])
                    
                    # Get price data up to this window
                    window_end = window_start + timedelta(hours=1)
                    price_subset = price_data[symbol][
                        (price_data[symbol]['Timestamp'] >= window_start) &
                        (price_data[symbol]['Timestamp'] < window_end)
                    ]
                    
                    if len(price_subset) > 0:
                        # Mock market sentiment (in real implementation, this would be calculated from market-wide news)
                        market_sentiment = 0.0
                        
                        # Generate signal
                        signal = self.signal_engine.decide(
                            avg_sentiment, market_sentiment, price_subset
                        )
                        
                        signals[symbol].append({
                            'timestamp': window_start,
                            'signal': signal
                        })
        
        return signals
    
    def _group_sentiment_by_time(self, sentiment_list: List[Dict]) -> Dict[datetime, List[Dict]]:
        """Group sentiment data by time windows"""
        windows = {}
        
        for sentiment in sentiment_list:
            # Round to nearest hour
            window_start = sentiment['timestamp'].replace(minute=0, second=0, microsecond=0)
            
            if window_start not in windows:
                windows[window_start] = []
            
            windows[window_start].append(sentiment)
        
        return windows
    
    def _execute_trades(self, signals: Dict[str, List[Dict]], price_data: Dict[str, pd.DataFrame]) -> List[Trade]:
        """Execute trades based on signals with multiple time horizons"""
        trades = []
        
        # Define multiple time horizons for PnL calculation
        time_horizons = [5, 15, 30, 60]  # minutes
        
        for symbol, signal_list in signals.items():
            if symbol not in price_data:
                continue
            
            for signal_info in signal_list:
                timestamp = signal_info['timestamp']
                signal = signal_info['signal']
                
                # Find price at signal time
                price_at_signal = price_data[symbol][
                    price_data[symbol]['Timestamp'] >= timestamp
                ]
                
                if len(price_at_signal) > 0:
                    entry_price = price_at_signal.iloc[0]['Close']
                    
                    # Create trade
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal=signal.signal,
                        entry_price=entry_price,
                        confidence=signal.confidence,
                        reasoning=signal.reasoning
                    )
                    
                    # Calculate PnL for multiple time horizons
                    pnl_results = {}
                    
                    for horizon_minutes in time_horizons:
                        exit_time = timestamp + timedelta(minutes=horizon_minutes)
                        exit_prices = price_data[symbol][
                            price_data[symbol]['Timestamp'] >= exit_time
                        ]
                        
                        if len(exit_prices) > 0:
                            exit_price = exit_prices.iloc[0]['Close']
                            
                            # Calculate PnL for this horizon
                            if signal.signal == SignalType.LONG:
                                pnl = (exit_price - entry_price) / entry_price
                            elif signal.signal == SignalType.SHORT:
                                pnl = (entry_price - exit_price) / entry_price
                            else:
                                pnl = 0.0
                            
                            pnl_results[f"{horizon_minutes}min"] = {
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'exit_time': exit_time
                            }
                    
                    # Use the primary horizon (60min) for main PnL
                    if "60min" in pnl_results:
                        trade.exit_price = pnl_results["60min"]['exit_price']
                        trade.pnl = pnl_results["60min"]['pnl']
                    elif pnl_results:
                        # Use the longest available horizon
                        longest_horizon = max(pnl_results.keys(), key=lambda x: int(x.replace('min', '')))
                        trade.exit_price = pnl_results[longest_horizon]['exit_price']
                        trade.pnl = pnl_results[longest_horizon]['pnl']
                    
                    # Store all horizon results in trade metadata
                    trade.metadata = {
                        'pnl_horizons': pnl_results,
                        'signal_strength': signal.confidence,
                        'relative_sentiment': getattr(signal, 'relative_sentiment', 0.0)
                    }
                    
                    trades.append(trade)
        
        return trades
    
    def _calculate_performance(self, trades: List[Trade]) -> BacktestResult:
        """Calculate performance metrics"""
        if not trades:
            return BacktestResult(
                trades=[],
                total_pnl=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                summary_stats={}
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
        
        # Summary statistics
        summary_stats = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t.pnl for t in trades if t.pnl and t.pnl < 0]) if any(t.pnl and t.pnl < 0 for t in trades) else 0.0,
            'profit_factor': sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0)) if any(t.pnl and t.pnl < 0 for t in trades) else float('inf')
        }
        
        return BacktestResult(
            trades=trades,
            total_pnl=total_pnl,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            summary_stats=summary_stats
        ) 