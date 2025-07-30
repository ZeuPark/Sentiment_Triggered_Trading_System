#!/usr/bin/env python3
"""
SentimentTriggeredTradingSystem - Main Execution File

실시간/과거 뉴스 감정 분석과 1분봉 시세를 활용한 수동 매매 보조 시스템
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetchers.news_fetcher import MockNewsFetcher, NewsAPIFetcher, FinnhubNewsFetcher
from fetchers.price_fetcher import MockPriceFetcher, YahooFinanceFetcher, TwelveDataFetcher
from analyzers.sentiment_analyzer import MockSentimentAnalyzer, VADERSentimentAnalyzer, FinBERTSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester
from visualizer import Visualizer

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
            price_fetcher = YahooFinanceFetcher()  # No API key required
        except Exception as e:
            print(f"⚠️  Yahoo Finance error: {e}, falling back to mock")
            price_fetcher = MockPriceFetcher()
        
        try:
            sentiment_analyzer = VADERSentimentAnalyzer()  # No API key required
        except Exception as e:
            print(f"⚠️  VADER sentiment error: {e}, falling back to mock")
            sentiment_analyzer = MockSentimentAnalyzer()
    
    # Signal engine (same for both mock and real)
    signal_engine = SignalEngine(
        sentiment_threshold=1.0,
        trend_lookback_periods=5,
        volume_threshold=1.5
    )
    
    return news_fetcher, price_fetcher, sentiment_analyzer, signal_engine

def run_backtest(symbols: List[str], 
                start_time: datetime, 
                end_time: datetime,
                use_mock: bool = True,
                visualize: bool = True,
                news_source: str = "auto") -> None:
    """
    Run backtest simulation
    
    Args:
        symbols: List of symbols to trade
        start_time: Start of backtest period
        end_time: End of backtest period
        use_mock: Whether to use mock data
        visualize: Whether to show visualizations
    """
    
    print(f"\n🚀 Starting Backtest Simulation")
    print(f"📊 Symbols: {symbols}")
    print(f"📅 Period: {start_time} to {end_time}")
    print(f"🔧 Mode: {'Mock' if use_mock else 'Real'} Data")
    
    # Setup components
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(use_mock)
    
    # Create backtester
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine,
        initial_capital=10000.0,
        position_size=0.1
    )
    
    # Run backtest
    print("\n📈 Running backtest...")
    result = backtester.run(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        interval='1m'
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"Total PnL: {result.total_pnl:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Average Trade PnL: {result.avg_trade_pnl:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    print(f"\n📋 Detailed Statistics:")
    for key, value in result.summary_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key or 'win' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Show trade details
    if result.trades:
        print(f"\n📝 Recent Trades:")
        for i, trade in enumerate(result.trades[-5:]):  # Show last 5 trades
            print(f"  {i+1}. {trade.timestamp} | {trade.symbol} | {trade.signal.value} | "
                  f"Entry: ${trade.entry_price:.2f} | Exit: ${trade.exit_price:.2f} | "
                  f"PnL: {trade.pnl:.2%} | Confidence: {trade.confidence:.2f}")
    
    # Visualize results
    if visualize and result.trades:
        print("\n📊 Generating visualizations...")
        visualizer = Visualizer(style='plotly')
        visualizer.plot_backtest_results(result)
        
        # Additional visualizations
        if hasattr(result, 'sentiment_data') and result.sentiment_data:
            print("📈 Generating sentiment heatmap...")
            visualizer.plot_sentiment_heatmap(result.sentiment_data)
        
        if hasattr(result, 'symbol_signals') and result.symbol_signals:
            print("📊 Generating market vs symbol analysis...")
            # Mock market signals for demonstration
            market_signals = {symbol: signals for symbol, signals in result.symbol_signals.items()}
            visualizer.plot_market_vs_symbol_signals(
                result.symbol_signals, 
                market_signals, 
                result.price_data
            )
    
    return result

def run_realtime_monitoring(symbols: List[str], 
                          use_mock: bool = True,
                          monitoring_duration: int = 60) -> None:
    """
    Run real-time monitoring mode
    
    Args:
        symbols: List of symbols to monitor
        use_mock: Whether to use mock data
        monitoring_duration: Duration in minutes to monitor
    """
    
    print(f"\n🔍 Starting Real-time Monitoring")
    print(f"📊 Symbols: {symbols}")
    print(f"⏱️  Duration: {monitoring_duration} minutes")
    print(f"🔧 Mode: {'Mock' if use_mock else 'Real'} Data")
    
    # Setup components
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(use_mock)
    
    # Monitoring loop
    end_time = datetime.now() + timedelta(minutes=monitoring_duration)
    
    print(f"\n🔄 Starting monitoring loop (until {end_time})...")
    print("Press Ctrl+C to stop early")
    
    try:
        while datetime.now() < end_time:
            current_time = datetime.now()
            
            print(f"\n⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
            
            # Fetch recent data (last hour)
            start_time = current_time - timedelta(hours=1)
            
            for symbol in symbols:
                try:
                    # Fetch recent news
                    news_data = news_fetcher.fetch_news(symbol, start_time, current_time)
                    
                    if news_data:
                        # Analyze sentiment
                        sentiments = []
                        for news in news_data:
                            sentiment = sentiment_analyzer.analyze(news['headline'])
                            sentiments.append(sentiment)
                        
                        avg_sentiment = sum(sentiments) / len(sentiments)
                        
                        # Fetch recent price data
                        price_data = price_fetcher.fetch_prices(symbol, '1m', start_time, current_time)
                        
                        if len(price_data) > 0:
                            # Generate signal
                            signal = signal_engine.decide(
                                symbol_sentiment=avg_sentiment,
                                market_sentiment=0.0,  # Mock market sentiment
                                price_df=price_data
                            )
                            
                            # Display results
                            print(f"📊 {symbol}:")
                            print(f"  💭 Sentiment: {avg_sentiment:.3f}")
                            print(f"  📈 Signal: {signal.signal.value}")
                            print(f"  🎯 Confidence: {signal.confidence:.2f}")
                            print(f"  💡 Reasoning: {signal.reasoning}")
                            print(f"  📰 Recent News: {len(news_data)} articles")
                        else:
                            print(f"📊 {symbol}: No price data available")
                    else:
                        print(f"📊 {symbol}: No recent news")
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
            
            # Wait before next iteration
            import time
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    print("\n✅ Monitoring completed")

def demo_mode() -> None:
    """Run demo mode with sample data"""
    
    print("🎮 DEMO MODE")
    print("="*50)
    
    # Sample symbols
    symbols = ['TSLA', 'NVDA', 'AAPL']
    
    # Sample time period (last 7 days)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
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
  python main.py --demo                    # Run demo mode
  python main.py --backtest TSLA NVDA     # Run backtest on specific symbols
  python main.py --monitor AAPL MSFT      # Run real-time monitoring
  python main.py --backtest TSLA --real   # Run backtest with real APIs
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
            visualize=not args.no_viz
        )
    
    elif args.monitor:
        run_realtime_monitoring(
            symbols=args.monitor,
            use_mock=not args.real,
            monitoring_duration=args.duration
        )
    
    else:
        # Default to demo mode
        print("No mode specified, running demo...")
        demo_mode()

if __name__ == "__main__":
    main() 