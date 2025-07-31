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
    """Permissive backtesting engine: triggers trades frequently, minimal filtering."""

    def __init__(self, 
                 news_fetcher: NewsFetcher,
                 price_fetcher: PriceFetcher,
                 sentiment_analyzer: SentimentAnalyzer,
                 signal_engine: SignalEngine,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.05,
                 max_wait_minutes: int = 30,
                 quick_entry_threshold: float = 0.005,
                 quick_entry_minutes: int = 3):
        self.news_fetcher = news_fetcher
        self.price_fetcher = price_fetcher
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_engine = signal_engine
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_wait_minutes = max_wait_minutes
        self.quick_entry_threshold = quick_entry_threshold
        self.quick_entry_minutes = quick_entry_minutes

    def run(self, 
            symbols: List[str],
            start_time: datetime,
            end_time: datetime,
            interval: str = '1m',
            cached_data: Dict = None) -> 'BacktestResult':
        print(f"Starting permissive backtest for {symbols} from {start_time} to {end_time}")

        news_data = self._fetch_news_data(symbols, start_time, end_time)
        price_data = self._fetch_price_data(symbols, interval, start_time, end_time)
        sentiment_data = self._process_sentiment(news_data)

        trades = []
        for symbol in symbols:
            if symbol not in price_data or price_data[symbol].empty:
                continue
            price_df = price_data[symbol]
            price_df = price_df.sort_values('Timestamp')
            sentiments = sentiment_data.get(symbol, [])
            # Map news by timestamp for fast lookup
            news_by_time = {n['timestamp']: n for n in sentiments}
            position = None  # {'type': 'LONG'/'SHORT', 'entry_price': float, 'entry_time': datetime}
            for idx, row in price_df.iterrows():
                current_time = row['Timestamp']
                current_price = row['Close']
                # Find latest news up to current_time
                relevant_news = [n for n in sentiments if n['timestamp'] <= current_time]
                if relevant_news:
                    latest_news = relevant_news[-1]
                    sentiment_score = latest_news['sentiment_score']
                else:
                    sentiment_score = 0
                signal = self.signal_engine.generate_signal(sentiment_score)
                # Allow multiple trades per day, flip position if signal changes
                if signal == "LONG":
                    if not position or position['type'] != "LONG":
                        # Close SHORT if open
                        if position:
                            trades.append(self._close_trade(symbol, position, current_time, current_price))
                        # Open LONG
                        position = {'type': "LONG", 'entry_price': current_price, 'entry_time': current_time}
                elif signal == "SHORT":
                    if not position or position['type'] != "SHORT":
                        # Close LONG if open
                        if position:
                            trades.append(self._close_trade(symbol, position, current_time, current_price))
                        # Open SHORT
                        position = {'type': "SHORT", 'entry_price': current_price, 'entry_time': current_time}
                else:  # HOLD
                    if position:
                        trades.append(self._close_trade(symbol, position, current_time, current_price))
                        position = None
                # No cooldown, no news count/confidence filter, always check every bar
            # Close any open position at end
            if position:
                trades.append(self._close_trade(symbol, position, price_df.iloc[-1]['Timestamp'], price_df.iloc[-1]['Close']))
        result = self._calculate_enhanced_performance(trades)
        return result

    def _fetch_news_data(self, symbols: List[str], start_time: datetime, end_time: datetime) -> Dict[str, List[Dict]]:
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
                            'confidence': 0.8  # Default confidence, not used for filtering
                        })
                except Exception as e:
                    print(f"Error processing sentiment: {e}")
                    continue
        return sentiment_data

    def _close_trade(self, symbol, position, exit_time, exit_price):
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        signal_type = SignalType.LONG if position['type'] == "LONG" else SignalType.SHORT
        pnl = (exit_price - entry_price) / entry_price if position['type'] == "LONG" else (entry_price - exit_price) / entry_price
        return Trade(
            timestamp=entry_time,
            symbol=symbol,
            signal=signal_type,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            confidence=1.0,
            reasoning="Permissive logic",
            exit_reason="SIGNAL_FLIP",
            hold_minutes=int((exit_time - entry_time).total_seconds() / 60)
        )

    def _calculate_enhanced_performance(self, trades: List[Trade]) -> 'BacktestResult':
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
        total_pnl = sum(trade.pnl or 0.0 for trade in trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        avg_trade_pnl = total_pnl / len(trades) if trades else 0.0
        cumulative_pnl = np.cumsum([t.pnl or 0.0 for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        returns = [t.pnl or 0.0 for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        strong_signals = [t for t in trades if t.signal == SignalType.LONG or t.signal == SignalType.SHORT]
        weak_signals = []
        strong_signal_pnl = sum(t.pnl or 0.0 for t in strong_signals) if strong_signals else 0.0
        weak_signal_pnl = 0.0
        stop_loss_count = len([t for t in trades if t.exit_reason == "STOP_LOSS"])
        take_profit_count = len([t for t in trades if t.exit_reason == "TAKE_PROFIT"])
        avg_hold_time = np.mean([t.hold_minutes for t in trades]) if trades else 0.0
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