#!/usr/bin/env python3
"""
Test script for enhanced signal generation system
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from fetchers.news_fetcher import NewsAPIFetcher
from fetchers.price_fetcher import AlpacaMarketFetcher
from analyzers.sentiment_analyzer import FinBERTSentimentAnalyzer
from engine.signal_engine import SignalEngine, SignalType
from backtest.backtester import Backtester

def test_enhanced_system():
    """Test the enhanced signal generation system"""
    
    print("🧪 Testing Enhanced Signal Generation System")
    print("=" * 50)
    
    # Setup components
    print("🔧 Setting up components...")
    news_fetcher = NewsAPIFetcher()
    price_fetcher = AlpacaMarketFetcher()
    sentiment_analyzer = FinBERTSentimentAnalyzer()
    
    # Test with more sensitive parameters
    signal_engine = SignalEngine(
        sentiment_threshold=0.2,  # Lower threshold for more signals
        news_confidence_threshold=0.4,  # Lower confidence requirement
        strong_signal_threshold=0.6,  # Lower strong signal threshold
        price_momentum_weight=0.3
    )
    
    # Test period
    start_time = datetime(2025, 7, 1, 9, 30)
    end_time = datetime(2025, 7, 1, 16, 0)  # Single day test
    
    print(f"📅 Test period: {start_time} to {end_time}")
    
    # Test with single symbol first
    symbol = 'TSLA'
    
    try:
        # Fetch data
        print(f"📊 Fetching data for {symbol}...")
        news = news_fetcher.fetch_news(symbol, start_time, end_time)
        prices = price_fetcher.fetch_prices(symbol, '1m', start_time, end_time)
        
        print(f"✅ Found {len(news)} news articles and {len(prices)} price records")
        
        # Process sentiment
        print("🧠 Processing sentiment...")
        sentiment_data = []
        for article in news[:10]:  # Test with first 10 articles
            try:
                sentiment_score = sentiment_analyzer.analyze(article['headline'])
                sentiment_data.append({
                    'timestamp': article['timestamp'],
                    'headline': article['headline'],
                    'sentiment_score': sentiment_score,
                    'confidence': 0.8
                })
                print(f"   {article['headline'][:50]}... → {sentiment_score:.3f}")
            except Exception as e:
                print(f"   Error processing: {e}")
                continue
        
        if sentiment_data:
            # Group by time windows
            print("⏰ Grouping sentiment by time...")
            sentiment_windows = {}
            for sentiment in sentiment_data:
                hour = sentiment['timestamp'].replace(minute=0, second=0, microsecond=0)
                if hour not in sentiment_windows:
                    sentiment_windows[hour] = []
                sentiment_windows[hour].append(sentiment)
            
            print(f"   Created {len(sentiment_windows)} time windows")
            
            # Test signal generation
            print("🎯 Testing signal generation...")
            signals_generated = 0
            
            for window_start, window_sentiments in sentiment_windows.items():
                if window_sentiments:
                    avg_sentiment = sum(s['sentiment_score'] for s in window_sentiments) / len(window_sentiments)
                    
                    # Get price data for this window
                    window_end = window_start + timedelta(hours=1)
                    price_subset = prices[
                        (prices['Timestamp'] >= window_start) &
                        (prices['Timestamp'] < window_end)
                    ]
                    
                    if len(price_subset) >= 5:
                        # Mock market sentiment
                        market_sentiment = 0.0
                        
                        # Calculate news confidence
                        news_confidence = min(1.0, len(window_sentiments) / 5.0)
                        
                        # Generate signal
                        signal = signal_engine.decide(
                            avg_sentiment, market_sentiment, price_subset, news_confidence
                        )
                        
                        print(f"   {window_start}: Sentiment={avg_sentiment:.3f}, Signal={signal.signal.value}, Score={signal.signal_score:.3f}")
                        
                        if signal.signal != SignalType.HOLD:
                            signals_generated += 1
            
            print(f"✅ Generated {signals_generated} non-HOLD signals")
            
            # Test backtester
            if signals_generated > 0:
                print("🔄 Testing backtester...")
                backtester = Backtester(
                    news_fetcher=news_fetcher,
                    price_fetcher=price_fetcher,
                    sentiment_analyzer=sentiment_analyzer,
                    signal_engine=signal_engine,
                    stop_loss_pct=0.03,
                    take_profit_pct=0.05
                )
                
                result = backtester.run([symbol], start_time, end_time, '1m')
                
                print(f"📊 Backtest Results:")
                print(f"   Total Trades: {result.total_trades}")
                print(f"   Total PnL: {result.total_pnl:.2%}")
                print(f"   Win Rate: {result.win_rate:.1%}")
                print(f"   Strong Signals: {result.summary_stats.get('strong_signals', 0)}")
                print(f"   Weak Signals: {result.summary_stats.get('weak_signals', 0)}")
                
                if result.trades:
                    print("   Sample trades:")
                    for i, trade in enumerate(result.trades[:3]):
                        print(f"     {i+1}. {trade.symbol} {trade.signal.value} - PnL: {trade.pnl:.2%}, Exit: {trade.exit_reason}")
            else:
                print("⚠️ No signals generated, skipping backtest")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_system() 