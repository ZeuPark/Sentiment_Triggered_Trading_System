#!/usr/bin/env python3
"""
Multi-Horizon PnL Analysis
Analyze trading results across different time horizons
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetchers.news_fetcher import MockNewsFetcher
from fetchers.price_fetcher import MockPriceFetcher
from analyzers.sentiment_analyzer import MockSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester

def run_multi_horizon_analysis():
    """Run backtest and analyze multi-horizon results"""
    
    # Setup components
    news_fetcher = MockNewsFetcher()
    price_fetcher = MockPriceFetcher()
    sentiment_analyzer = MockSentimentAnalyzer()
    signal_engine = SignalEngine()
    
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine
    )
    
    # Run backtest
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    result = backtester.run(
        symbols=['TSLA'],
        start_time=start_time,
        end_time=end_time,
        interval='1m'
    )
    
    return result

def analyze_horizon_results(result):
    """Analyze multi-horizon PnL results"""
    
    print("📊 Multi-Horizon PnL Analysis")
    print("=" * 50)
    
    if not result.trades:
        print("❌ No trades found")
        return
    
    # Extract horizon data
    horizon_data = []
    for trade in result.trades:
        if trade.metadata and 'pnl_horizons' in trade.metadata:
            for horizon, data in trade.metadata['pnl_horizons'].items():
                horizon_data.append({
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'signal': trade.signal.value,
                    'horizon': horizon,
                    'pnl': data['pnl'],
                    'exit_price': data['exit_price'],
                    'confidence': trade.confidence,
                    'relative_sentiment': trade.metadata.get('relative_sentiment', 0.0)
                })
    
    if not horizon_data:
        print("❌ No horizon data found")
        return
    
    df_horizons = pd.DataFrame(horizon_data)
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"Total Trades: {len(result.trades)}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Total PnL: {result.total_pnl:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    # Horizon analysis
    print(f"\n⏰ Horizon Analysis:")
    horizon_summary = df_horizons.groupby('horizon').agg({
        'pnl': ['mean', 'std', 'count'],
        'signal': lambda x: x.value_counts().to_dict()
    }).round(4)
    
    print(horizon_summary)
    
    # Signal analysis
    print(f"\n🎯 Signal Analysis:")
    signal_summary = df_horizons.groupby('signal').agg({
        'pnl': ['mean', 'count'],
        'confidence': 'mean'
    }).round(4)
    
    print(signal_summary)
    
    # Sentiment correlation
    print(f"\n🧠 Sentiment Correlation:")
    sentiment_corr = df_horizons.groupby('relative_sentiment').agg({
        'pnl': 'mean'
    }).round(4)
    
    print(sentiment_corr)
    
    return df_horizons

def create_horizon_visualizations(df_horizons):
    """Create visualizations for horizon analysis"""
    
    # 1. Horizon comparison box plot
    fig1 = px.box(df_horizons, x='horizon', y='pnl', 
                  title="PnL Distribution by Time Horizon",
                  labels={'pnl': 'PnL (%)', 'horizon': 'Time Horizon'})
    fig1.show()
    
    # 2. Signal performance
    fig2 = px.bar(df_horizons.groupby('signal')['pnl'].mean().reset_index(),
                  x='signal', y='pnl',
                  title="Average PnL by Signal Type",
                  labels={'pnl': 'Average PnL (%)', 'signal': 'Signal Type'})
    fig2.show()
    
    # 3. Sentiment vs PnL scatter
    fig3 = px.scatter(df_horizons, x='relative_sentiment', y='pnl',
                      color='horizon',
                      title="Sentiment Score vs PnL by Horizon",
                      labels={'relative_sentiment': 'Relative Sentiment', 'pnl': 'PnL (%)'})
    fig3.show()
    
    # 4. Confidence vs PnL
    fig4 = px.scatter(df_horizons, x='confidence', y='pnl',
                      color='signal',
                      title="Confidence vs PnL by Signal Type",
                      labels={'confidence': 'Signal Confidence', 'pnl': 'PnL (%)'})
    fig4.show()

def main():
    """Main execution"""
    print("🚀 Running Multi-Horizon Analysis...")
    
    # Run backtest
    result = run_multi_horizon_analysis()
    
    # Analyze results
    df_horizons = analyze_horizon_results(result)
    
    if df_horizons is not None:
        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_horizon_visualizations(df_horizons)
        
        # Save results
        df_horizons.to_csv('multi_horizon_results.csv', index=False)
        print("✅ Results saved to 'multi_horizon_results.csv'")

if __name__ == "__main__":
    main() 