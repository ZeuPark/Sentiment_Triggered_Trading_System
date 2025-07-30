#!/usr/bin/env python3
"""
SentimentTriggeredTradingSystem - Main Execution File

실시간/과거 뉴스 감정 분석과 1분봉 시세를 활용한 수동 매매 보조 시스템
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetchers.news_fetcher import MockNewsFetcher, NewsAPIFetcher, FinnhubNewsFetcher
from fetchers.price_fetcher import MockPriceFetcher, AlpacaMarketFetcher, TwelveDataFetcher
from analyzers.sentiment_analyzer import MockSentimentAnalyzer, VADERSentimentAnalyzer, FinBERTSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester
from visualizer import Visualizer

# Data caching
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(data_type: str, symbol: str, start_date: str, end_date: str) -> str:
    """Generate cache file path"""
    return os.path.join(CACHE_DIR, f"{data_type}_{symbol}_{start_date}_{end_date}.pkl")

def load_cached_data(cache_path: str):
    """Load cached data if exists and not expired"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Check if cache is not older than 24 hours
                if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(hours=24):
                    print(f"📦 Using cached data: {cache_path}")
                    return cached_data.get('data')
        except Exception as e:
            print(f"⚠️  Cache loading failed: {e}")
    return None

def save_cached_data(cache_path: str, data):
    """Save data to cache"""
    try:
        cached_data = {
            'data': data,
            'timestamp': datetime.now()
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"💾 Data cached: {cache_path}")
    except Exception as e:
        print(f"⚠️  Cache saving failed: {e}")

def setup_components(use_mock: bool = True, news_source: str = "auto") -> tuple:
    """
    Setup all system components
    
    Args:
        use_mock: Whether to use mock data for testing
        news_source: News source preference ("auto", "finnhub", "newsapi")
        
    Returns:
        Tuple of (news_fetcher, price_fetcher, sentiment_analyzer, signal_engine)
    """
    
    if use_mock:
        print("🔧 Setting up components with MOCK data for testing...")
        
        # Mock components for testing
        news_fetcher = MockNewsFetcher()
        price_fetcher = MockPriceFetcher()
        sentiment_analyzer = MockSentimentAnalyzer()
        
    else:
        print("🔧 Setting up components with REAL APIs...")
        
        # Real components (requires API keys)
        if news_source == "finnhub":
            try:
                news_fetcher = FinnhubNewsFetcher()  # Requires FINNHUB_API_KEY
                print("📰 Using Finnhub for financial news")
            except ValueError:
                print("⚠️  Finnhub API key not found, falling back to NewsAPI")
                news_source = "newsapi"
        
        if news_source in ["auto", "newsapi"]:
            try:
                news_fetcher = NewsAPIFetcher()  # Requires NEWS_API_KEY
                print("📰 Using NewsAPI for general news")
            except ValueError:
                print("⚠️  NewsAPI key not found, falling back to mock")
                news_fetcher = MockNewsFetcher()
        
        try:
            price_fetcher = AlpacaMarketFetcher()  # Requires ALPACA_API_KEY and ALPACA_SECRET_KEY
            print("📈 Using Alpaca Market API for price data")
        except ValueError as e:
            print(f"⚠️  Alpaca API key not found: {e}, falling back to mock")
            price_fetcher = MockPriceFetcher()
        except Exception as e:
            print(f"⚠️  Alpaca API error: {e}, falling back to mock")
            price_fetcher = MockPriceFetcher()
        
        try:
            sentiment_analyzer = FinBERTSentimentAnalyzer()  # Financial-specific BERT
            print("🧠 Using FinBERT for financial sentiment analysis")
        except Exception as e:
            print(f"⚠️  FinBERT error: {e}, falling back to VADER")
            try:
                sentiment_analyzer = VADERSentimentAnalyzer()  # No API key required
            except Exception as e2:
                print(f"⚠️  VADER sentiment error: {e2}, falling back to mock")
                sentiment_analyzer = MockSentimentAnalyzer()
    
    # Initialize components with enhanced parameters
    signal_engine = SignalEngine(
        sentiment_threshold=0.1,  # Even lower threshold for more signals
        news_confidence_threshold=0.2,  # Lower confidence requirement
        strong_signal_threshold=0.3,  # Lower strong signal threshold
        price_momentum_weight=0.3
    )
    
    return news_fetcher, price_fetcher, sentiment_analyzer, signal_engine

def run_backtest(symbols: List[str], 
                start_time: datetime, 
                end_time: datetime,
                use_mock: bool = True,
                visualize: bool = True,
                news_source: str = "auto",
                use_cache: bool = True) -> None:
    """
    Run backtest simulation with caching support
    
    Args:
        symbols: List of symbols to trade
        start_time: Start of backtest period
        end_time: End of backtest period
        use_mock: Whether to use mock data
        visualize: Whether to show visualizations
        news_source: News source preference
        use_cache: Whether to use data caching
    """
    
    print(f"\n🚀 Starting backtest for {symbols}")
    print(f"📅 Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print(f"🔧 Mode: {'Mock' if use_mock else 'Real'} data")
    
    # Setup components
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(use_mock, news_source)
    
    # Create backtester
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine,
        stop_loss_pct=0.03,  # 3% stop loss
        take_profit_pct=0.05,  # 5% take profit
        max_wait_minutes=15,  # Shorter wait time
        quick_entry_threshold=0.003,  # 0.3% for quick entry
        quick_entry_minutes=2
    )
    
    # Check cache for each symbol
    cached_data = {}
    if use_cache and not use_mock:
        for symbol in symbols:
            cache_path = get_cache_path("backtest", symbol, 
                                      start_time.strftime('%Y%m%d'), 
                                      end_time.strftime('%Y%m%d'))
            cached = load_cached_data(cache_path)
            if cached:
                cached_data[symbol] = cached
    
    # Run backtest
    try:
        result = backtester.run(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval='1m',
            cached_data=cached_data
        )
        
        # Cache results if not using mock data
        if use_cache and not use_mock:
            for symbol in symbols:
                cache_path = get_cache_path("backtest", symbol, 
                                          start_time.strftime('%Y%m%d'), 
                                          end_time.strftime('%Y%m%d'))
                save_cached_data(cache_path, result)
        
        # Print results
        print(f"\n📊 Backtest Results:")
        print(f"Total PnL: {result.total_pnl:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Avg Trade PnL: {result.avg_trade_pnl:.2%}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        # Visualize if requested
        if visualize:
            visualizer = Visualizer(style='plotly')
            visualizer.plot_backtest_results(result)
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise

def run_realtime_monitoring(symbols: List[str], 
                          use_mock: bool = True,
                          monitoring_duration: int = 60,
                          news_source: str = "auto") -> None:
    """
    Run real-time monitoring
    
    Args:
        symbols: List of symbols to monitor
        use_mock: Whether to use mock data
        monitoring_duration: Duration in minutes
        news_source: News source preference
    """
    
    print(f"\n🔍 Starting real-time monitoring for {symbols}")
    print(f"⏱️  Duration: {monitoring_duration} minutes")
    print(f"🔧 Mode: {'Mock' if use_mock else 'Real'} data")
    
    # Setup components
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(use_mock, news_source)
    
    # Create backtester for monitoring
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine
    )
    
    # Monitor for specified duration
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=monitoring_duration)
    
    try:
        result = backtester.run(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            interval='1m'
        )
        
        print(f"\n📊 Monitoring Results:")
        print(f"Total PnL: {result.total_pnl:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        raise

def demo_mode() -> None:
    """Run demo mode with sample data"""
    
    print("\n🎮 Running Demo Mode")
    print("="*30)
    
    # Demo parameters
    symbols = ["TSLA", "AAPL"]
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    print(f"📊 Demo symbols: {symbols}")
    print(f"📅 Demo period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Run backtest with mock data
    result = run_backtest(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        use_mock=True,
        visualize=True
    )
    
    print("\n🎉 Demo completed!")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="SentimentTriggeredTradingSystem - AI-powered trading framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                                    # Run demo mode
  python main.py --backtest TSLA NVDA                     # Run backtest on specific symbols
  python main.py --backtest TSLA --start-date 2024-01-01 --end-date 2024-01-31  # Custom date range
  python main.py --monitor AAPL MSFT                      # Run real-time monitoring
  python main.py --backtest TSLA --real                   # Run backtest with real APIs
  python main.py --backtest TSLA --no-cache               # Disable data caching
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with sample data')
    
    parser.add_argument('--backtest', nargs='+', metavar='SYMBOL',
                       help='Run backtest on specified symbols')
    
    parser.add_argument('--monitor', nargs='+', metavar='SYMBOL',
                       help='Run real-time monitoring on specified symbols')
    
    parser.add_argument('--real', action='store_true',
                       help='Use real APIs instead of mock data')
    
    parser.add_argument('--start-date', type=str,
                       help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                       help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in minutes (default: 60)')
    
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    
    parser.add_argument('--news-source', choices=['auto', 'finnhub', 'newsapi'],
                       default='auto', help='Preferred news source (default: auto)')
    
    args = parser.parse_args()
    
    # Print banner
    print("🎯 SentimentTriggeredTradingSystem")
    print("="*50)
    print("AI-powered trading framework based on sentiment analysis")
    print("Built with OOP architecture for scalability and modularity")
    print("="*50)
    
    # Determine mode
    if args.demo:
        demo_mode()
    
    elif args.backtest:
        # Parse dates
        if args.start_date and args.end_date:
            start_time = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_time = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            # Default to last 7 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
        
        run_backtest(
            symbols=args.backtest,
            start_time=start_time,
            end_time=end_time,
            use_mock=not args.real,
            visualize=not args.no_viz,
            news_source=args.news_source,
            use_cache=not args.no_cache
        )
    
    elif args.monitor:
        run_realtime_monitoring(
            symbols=args.monitor,
            use_mock=not args.real,
            monitoring_duration=args.duration,
            news_source=args.news_source
        )
    
    else:
        # Default to demo mode
        print("No mode specified, running demo...")
        demo_mode()

if __name__ == "__main__":
    main() 