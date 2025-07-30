#!/usr/bin/env python3
"""
Parameter Optimization Baseline Run Only
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from fetchers.news_fetcher import NewsAPIFetcher
from fetchers.price_fetcher import AlpacaMarketFetcher
from analyzers.sentiment_analyzer import FinBERTSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester


def main():
    print("🚀 Baseline Backtest Run (No Parameter Search)")
    print("=" * 60)
    
    # Setup components
    news_fetcher = NewsAPIFetcher()
    price_fetcher = AlpacaMarketFetcher()
    sentiment_analyzer = FinBERTSentimentAnalyzer()
    signal_engine = SignalEngine(sentiment_threshold=0.15)
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine,
        stop_loss_pct=0.03,
        take_profit_pct=0.05
    )
    
    symbols = ['TSLA', 'NVDA']
    start_time = datetime(2025, 6, 29, 9, 30)
    end_time = datetime(2025, 7, 1, 16, 0)
    
    result = backtester.run(symbols, start_time, end_time, '1m')
    
    print("\n📊 Baseline Backtest Results:")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Total PnL: {result.total_pnl:.2%}")
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Avg Trade PnL: {result.avg_trade_pnl:.2%}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    if result.trades:
        print("   Sample trades:")
        for i, trade in enumerate(result.trades[:3]):
            print(f"     {i+1}. {trade.symbol} {trade.signal.value} - PnL: {trade.pnl:.2%}, Exit: {trade.exit_reason}")
    else:
        print("   No trades executed.")

if __name__ == "__main__":
    main() 