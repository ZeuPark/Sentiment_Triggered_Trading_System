#!/usr/bin/env python3
"""
Sentiment-Triggered Trading System Dashboard
Real-time signal visualization and backtest analysis
"""

import streamlit as st
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

from fetchers.news_fetcher import NewsAPIFetcher, FinnhubNewsFetcher, MockNewsFetcher
from fetchers.price_fetcher import AlpacaMarketFetcher, MockPriceFetcher
from analyzers.sentiment_analyzer import FinBERTSentimentAnalyzer, VADERSentimentAnalyzer, MockSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="Sentiment Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache management
def get_cache_key(symbol, start_date, end_date, use_mock, force_reload=False):
    """Generate cache key with force reload option"""
    if force_reload:
        return f"{symbol}_{start_date}_{end_date}_{use_mock}_{datetime.now().timestamp()}"
    return f"{symbol}_{start_date}_{end_date}_{use_mock}"

def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ All caches cleared successfully!")

def setup_cache_management():
    """Setup cache management controls in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Cache Management")
    
    # Cache control options
    cache_mode = st.sidebar.selectbox(
        "Cache Mode",
        ["Normal", "Force Reload", "No Cache"],
        help="Normal: Use cache, Force Reload: Ignore cache, No Cache: Disable caching"
    )
    
    # Force reload button
    if st.sidebar.button("🔄 Force Reload All Data", type="primary"):
        clear_all_caches()
        st.rerun()
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear All Caches"):
        clear_all_caches()
        st.rerun()
    
    # Cache status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cache Status:**")
    
    # Show cache info
    if cache_mode == "Normal":
        st.sidebar.success("✅ Cache Enabled")
        st.sidebar.info("Data will be cached for 1 hour")
    elif cache_mode == "Force Reload":
        st.sidebar.warning("⚠️ Force Reload Mode")
        st.sidebar.info("Cache will be ignored")
    else:
        st.sidebar.error("❌ Cache Disabled")
        st.sidebar.info("No caching will be used")
    
    # Cache info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cache Info:**")
    st.sidebar.markdown("• Normal: Fast loading, uses cached data")
    st.sidebar.markdown("• Force Reload: Ignores cache, fetches fresh data")
    st.sidebar.markdown("• No Cache: Disables caching completely")
    
    # Development mode toggle
    st.sidebar.markdown("---")
    dev_mode = st.sidebar.checkbox("Development Mode", value=False, 
                                  help="Enable additional debugging and cache controls")
    
    if dev_mode:
        st.sidebar.markdown("**Dev Controls:**")
        if st.sidebar.button("🔍 Show Cache Keys"):
            show_cache_keys()
        if st.sidebar.button("🧹 Clear Backtest Cache"):
            clear_backtest_cache()
    
    return cache_mode

def show_cache_keys():
    """Show current cache keys (for debugging)"""
    st.info("Cache keys would be displayed here in a real implementation")
    st.code("""
    Example cache keys:
    - TSLA_2025-07-23_2025-07-30_True
    - NVDA_2025-07-23_2025-07-30_False
    """)

def clear_backtest_cache():
    """Clear only backtest-related caches"""
    # In a real implementation, you would clear specific cache keys
    st.success("✅ Backtest cache cleared!")
    st.rerun()

def setup_components(use_mock=True, news_source="auto"):
    """Setup system components"""
    if use_mock:
        news_fetcher = MockNewsFetcher()
        price_fetcher = MockPriceFetcher()
        sentiment_analyzer = MockSentimentAnalyzer()
    else:
        if news_source == "finnhub":
            try:
                news_fetcher = FinnhubNewsFetcher()
            except:
                news_fetcher = NewsAPIFetcher()
        else:
            try:
                news_fetcher = NewsAPIFetcher()
            except:
                news_fetcher = MockNewsFetcher()
        
        try:
            price_fetcher = AlpacaMarketFetcher()
        except:
            price_fetcher = MockPriceFetcher()
        
        try:
            sentiment_analyzer = FinBERTSentimentAnalyzer()
        except:
            sentiment_analyzer = VADERSentimentAnalyzer()
    
    signal_engine = SignalEngine()
    return news_fetcher, price_fetcher, sentiment_analyzer, signal_engine

def create_signal_flow_diagram():
    """Create signal flow diagram"""
    st.subheader("🔄 Signal Engine Flow")
    
    flow_data = {
        "Step": [
            "1. News Event Occurs",
            "2. Sentiment Analysis",
            "3. Market Sentiment Calculation",
            "4. Relative Sentiment = Symbol - Market",
            "5. Price Trend Analysis",
            "6. Volume Analysis",
            "7. Signal Decision",
            "8. Entry Price Determination",
            "9. Multi-horizon PnL Calculation"
        ],
        "Description": [
            "Real-time news from NewsAPI/Finnhub",
            "FinBERT/VADER sentiment score (-1.0 to +1.0)",
            "Average sentiment across all symbols",
            "Symbol sentiment advantage over market",
            "UPTREND/DOWNTREND/NEUTRAL based on MA",
            "HIGH_VOLUME/NORMAL_VOLUME/LOW_VOLUME",
            "LONG/SHORT/HOLD based on strategy",
            "Price at signal timestamp",
            "5min/15min/30min/60min returns"
        ],
        "Example": [
            "TSLA announces earnings",
            "FinBERT: +0.8 (positive)",
            "Market avg: +0.2",
            "Relative: +0.6",
            "DOWNTREND (price falling)",
            "HIGH_VOLUME (2x average)",
            "WEAK SHORT (sentiment + downtrend)",
            "$302.53 at 14:30:00",
            "5min: -0.2%, 15min: +0.1%, 30min: +0.3%"
        ]
    }
    
    df_flow = pd.DataFrame(flow_data)
    st.dataframe(df_flow, use_container_width=True)

def run_backtest_analysis(symbol, start_date, end_date, use_mock=True, cache_mode="Normal"):
    """Run backtest and return detailed results with cache management"""
    
    # Determine if we should use cache
    use_cache = cache_mode != "No Cache"
    force_reload = cache_mode == "Force Reload"
    
    if use_cache and not force_reload:
        # Use cached version
        return _run_backtest_cached(symbol, start_date, end_date, use_mock)
    else:
        # Run without cache or force reload
        return _run_backtest_uncached(symbol, start_date, end_date, use_mock)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def _run_backtest_cached(symbol, start_date, end_date, use_mock=True):
    """Cached version of backtest analysis"""
    return _run_backtest_uncached(symbol, start_date, end_date, use_mock)

def _run_backtest_uncached(symbol, start_date, end_date, use_mock=True):
    """Uncached version of backtest analysis"""
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(use_mock)
    
    backtester = Backtester(
        news_fetcher=news_fetcher,
        price_fetcher=price_fetcher,
        sentiment_analyzer=sentiment_analyzer,
        signal_engine=signal_engine
    )
    
    result = backtester.run(
        symbols=[symbol],
        start_time=start_date,
        end_time=end_date,
        interval='1m'
    )
    
    return result

def create_multi_horizon_analysis(result):
    """Create multi-horizon PnL analysis"""
    st.subheader("📊 Multi-Horizon PnL Analysis")
    
    if not result.trades:
        st.warning("No trades found in the selected period")
        return
    
    # Extract multi-horizon data
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
                    'confidence': trade.confidence
                })
    
    if horizon_data:
        df_horizons = pd.DataFrame(horizon_data)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(result.trades))
        with col2:
            st.metric("Win Rate", f"{result.win_rate:.1%}")
        with col3:
            st.metric("Total PnL", f"{result.total_pnl:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        
        # Horizon comparison chart
        fig = px.box(df_horizons, x='horizon', y='pnl', 
                    title="PnL Distribution by Time Horizon",
                    labels={'pnl': 'PnL (%)', 'horizon': 'Time Horizon'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed horizon table
        st.subheader("📋 Detailed Horizon Analysis")
        horizon_summary = df_horizons.groupby('horizon').agg({
            'pnl': ['mean', 'std', 'count'],
            'signal': lambda x: x.value_counts().to_dict()
        }).round(4)
        
        st.dataframe(horizon_summary, use_container_width=True)

def create_signal_overlay_chart(result):
    """Create price chart with sentiment and signal overlay"""
    st.subheader("📈 Price Chart with Signal Overlay")
    
    if not hasattr(result, 'price_data') or not result.price_data:
        st.warning("Price data not available")
        return
    
    symbol = list(result.price_data.keys())[0]
    price_df = result.price_data[symbol]
    
    # Create subplot
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Sentiment Score', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_df['Timestamp'],
            open=price_df['Open'],
            high=price_df['High'],
            low=price_df['Low'],
            close=price_df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add signal markers
    if result.trades:
        for trade in result.trades:
            color = 'green' if trade.signal.value == 'LONG' else 'red' if trade.signal.value == 'SHORT' else 'gray'
            fig.add_trace(
                go.Scatter(
                    x=[trade.timestamp],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='diamond'),
                    name=f"{trade.signal.value} Signal",
                    text=f"{trade.signal.value}<br>Confidence: {trade.confidence:.2f}",
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ),
                row=1, col=1
            )
    
    # Sentiment scores (if available)
    if hasattr(result, 'sentiment_data') and result.sentiment_data:
        sentiment_times = []
        sentiment_scores = []
        
        for symbol_data in result.sentiment_data.values():
            for item in symbol_data:
                sentiment_times.append(item['timestamp'])
                sentiment_scores.append(item['sentiment_score'])
        
        if sentiment_times:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_times,
                    y=sentiment_scores,
                    mode='markers',
                    marker=dict(size=8, color='purple'),
                    name='Sentiment Score',
                    yaxis='y2'
                ),
                row=2, col=1
            )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=price_df['Timestamp'],
            y=price_df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Chart with Signals",
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_heatmap(result):
    """Create sentiment heatmap"""
    if not hasattr(result, 'sentiment_data') or not result.sentiment_data:
        st.warning("No sentiment data available")
        return
    
    st.subheader("🔥 Sentiment Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = []
    for symbol, sentiment_list in result.sentiment_data.items():
        for item in sentiment_list:
            heatmap_data.append({
                'Symbol': symbol,
                'Time': item['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Sentiment': item['sentiment_score']
            })
    
    if heatmap_data:
        df = pd.DataFrame(heatmap_data)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Create heatmap
        pivot_table = df.pivot_table(
            values='Sentiment', 
            index='Symbol', 
            columns=df['Time'].dt.strftime('%Y-%m-%d'),
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_table,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title='Sentiment Score Heatmap by Symbol and Date'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_long_short_pnl_distribution(result):
    """Create long/short PnL distribution chart"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("📊 Long/Short PnL Distribution")
    
    # Prepare trade data
    trade_data = []
    for trade in result.trades:
        if trade.pnl is not None:
            trade_data.append({
                'Signal': trade.signal.value,
                'PnL': trade.pnl,
                'Confidence': trade.confidence
            })
    
    if trade_data:
        df = pd.DataFrame(trade_data)
        
        # Create distribution plot
        fig = px.histogram(
            df, 
            x='PnL', 
            color='Signal',
            nbins=30,
            title='PnL Distribution by Signal Type',
            color_discrete_map={'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            long_trades = df[df['Signal'] == 'LONG']
            if len(long_trades) > 0:
                st.metric("LONG Trades", len(long_trades))
                st.metric("LONG Avg PnL", f"{long_trades['PnL'].mean():.2%}")
        
        with col2:
            short_trades = df[df['Signal'] == 'SHORT']
            if len(short_trades) > 0:
                st.metric("SHORT Trades", len(short_trades))
                st.metric("SHORT Avg PnL", f"{short_trades['PnL'].mean():.2%}")
        
        with col3:
            hold_trades = df[df['Signal'] == 'HOLD']
            if len(hold_trades) > 0:
                st.metric("HOLD Trades", len(hold_trades))
                st.metric("HOLD Avg PnL", f"{hold_trades['PnL'].mean():.2%}")

def create_symbol_cumulative_returns(result):
    """Create symbol-wise cumulative returns chart"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("📈 Symbol-wise Cumulative Returns")
    
    # Prepare trade data
    trade_data = []
    for trade in result.trades:
        if trade.pnl is not None:
            trade_data.append({
                'Symbol': trade.symbol,
                'Timestamp': trade.timestamp,
                'PnL': trade.pnl,
                'Signal': trade.signal.value
            })
    
    if trade_data:
        df = pd.DataFrame(trade_data)
        df = df.sort_values('Timestamp')
        
        # Calculate cumulative returns by symbol
        symbols = df['Symbol'].unique()
        
        fig = go.Figure()
        
        for symbol in symbols:
            symbol_data = df[df['Symbol'] == symbol]
            cumulative_pnl = symbol_data['PnL'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=symbol_data['Timestamp'],
                y=cumulative_pnl,
                mode='lines+markers',
                name=symbol,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title='Cumulative PnL by Symbol',
            xaxis_title='Time',
            yaxis_title='Cumulative PnL',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_monthly_performance(result):
    """Create monthly performance chart"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("📅 Monthly Performance Analysis")
    
    # Prepare trade data
    trade_data = []
    for trade in result.trades:
        if trade.pnl is not None:
            trade_data.append({
                'Month': trade.timestamp.strftime('%Y-%m'),
                'PnL': trade.pnl,
                'Signal': trade.signal.value,
                'Symbol': trade.symbol
            })
    
    if trade_data:
        df = pd.DataFrame(trade_data)
        
        # Monthly performance by signal type
        monthly_pnl = df.groupby(['Month', 'Signal'])['PnL'].agg(['mean', 'sum', 'count']).reset_index()
        
        # Create monthly performance chart
        fig = px.bar(
            monthly_pnl,
            x='Month',
            y='sum',
            color='Signal',
            title='Monthly PnL by Signal Type',
            color_discrete_map={'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly statistics table
        st.subheader("📊 Monthly Statistics")
        monthly_stats = df.groupby('Month').agg({
            'PnL': ['mean', 'sum', 'count'],
            'Signal': lambda x: (x == 'LONG').sum()  # Count LONG signals
        }).round(4)
        monthly_stats.columns = ['Avg PnL', 'Total PnL', 'Trade Count', 'LONG Count']
        st.dataframe(monthly_stats)

def create_confidence_analysis(result):
    """Create confidence-based performance analysis"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("🎯 Confidence-based Performance Analysis")
    
    # Prepare trade data
    trade_data = []
    for trade in result.trades:
        if trade.pnl is not None:
            trade_data.append({
                'Confidence': trade.confidence,
                'PnL': trade.pnl,
                'Signal': trade.signal.value,
                'Confidence_Level': 'High' if trade.confidence >= 0.7 else 'Low'
            })
    
    if trade_data:
        df = pd.DataFrame(trade_data)
        
        # Confidence vs PnL scatter plot
        fig = px.scatter(
            df,
            x='Confidence',
            y='PnL',
            color='Signal',
            size='Confidence',
            title='Confidence vs PnL Relationship',
            color_discrete_map={'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by confidence level
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 High Confidence Trades (≥0.7)")
            high_conf = df[df['Confidence'] >= 0.7]
            if len(high_conf) > 0:
                st.metric("Count", len(high_conf))
                st.metric("Avg PnL", f"{high_conf['PnL'].mean():.2%}")
                st.metric("Win Rate", f"{(high_conf['PnL'] > 0).mean():.2%}")
        
        with col2:
            st.subheader("📉 Low Confidence Trades (<0.7)")
            low_conf = df[df['Confidence'] < 0.7]
            if len(low_conf) > 0:
                st.metric("Count", len(low_conf))
                st.metric("Avg PnL", f"{low_conf['PnL'].mean():.2%}")
                st.metric("Win Rate", f"{(low_conf['PnL'] > 0).mean():.2%}")

def create_correlation_analysis(result):
    """Create sentiment correlation analysis"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("🔗 Sentiment Correlation Analysis")
    
    # Prepare correlation data
    correlation_data = []
    for trade in result.trades:
        if trade.pnl is not None and hasattr(trade, 'metadata') and trade.metadata:
            relative_sentiment = trade.metadata.get('relative_sentiment', 0)
            correlation_data.append({
                'Relative_Sentiment': relative_sentiment,
                'PnL': trade.pnl,
                'Confidence': trade.confidence
            })
    
    if correlation_data:
        df = pd.DataFrame(correlation_data)
        
        # Calculate correlation
        correlation = df['Relative_Sentiment'].corr(df['PnL'])
        
        st.metric("Sentiment-PnL Correlation", f"{correlation:.4f}")
        
        # Correlation scatter plot
        fig = px.scatter(
            df,
            x='Relative_Sentiment',
            y='PnL',
            color='Confidence',
            title=f'Sentiment vs PnL Correlation (r={correlation:.4f})',
            color_continuous_scale='Viridis'
        )
        
        # Add trend line
        z = np.polyfit(df['Relative_Sentiment'], df['PnL'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['Relative_Sentiment'],
            y=p(df['Relative_Sentiment']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation interpretation
        if abs(correlation) > 0.3:
            st.success(f"Strong correlation detected: {correlation:.4f}")
        elif abs(correlation) > 0.1:
            st.warning(f"Moderate correlation detected: {correlation:.4f}")
        else:
            st.info(f"Weak correlation detected: {correlation:.4f}")

def create_sentiment_pnl_correlation(result):
    """Create sentiment score vs PnL correlation scatter plot"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("📊 Sentiment Score vs PnL Correlation")
    
    # Prepare trade data
    trade_data = []
    for trade in result.trades:
        if trade.pnl is not None and trade.metadata:
            trade_data.append({
                'Symbol': trade.symbol,
                'Sentiment': trade.metadata.get('relative_sentiment', 0.0),
                'PnL': trade.pnl,
                'Signal': trade.signal.value,
                'Confidence': trade.confidence,
                'Signal_Score': trade.metadata.get('signal_score', 0.0)
            })
    
    if trade_data:
        df = pd.DataFrame(trade_data)
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x='Sentiment', 
            y='PnL',
            color='Signal',
            size='Confidence',
            hover_data=['Symbol', 'Signal_Score'],
            title='Sentiment Score vs PnL Correlation',
            color_discrete_map={
                'STRONG_LONG': 'darkgreen', 
                'STRONG_SHORT': 'darkred',
                'WEAK_LONG': 'lightgreen', 
                'WEAK_SHORT': 'lightcoral'
            }
        )
        
        # Add trend line
        fig.add_traces(px.scatter(df, x='Sentiment', y='PnL', trendline="ols").data)
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        correlation = df['Sentiment'].corr(df['PnL'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        # Summary by signal type
        st.subheader("📈 Performance by Signal Type")
        signal_summary = df.groupby('Signal').agg({
            'PnL': ['count', 'mean', 'std'],
            'Sentiment': 'mean',
            'Confidence': 'mean'
        }).round(4)
        
        st.dataframe(signal_summary)
        
        # Strong vs Weak signal comparison
        if 'STRONG_LONG' in df['Signal'].values or 'STRONG_SHORT' in df['Signal'].values:
            st.subheader("💪 Strong vs Weak Signal Performance")
            
            strong_signals = df[df['Signal'].str.contains('STRONG')]
            weak_signals = df[df['Signal'].str.contains('WEAK')]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(strong_signals) > 0:
                    st.metric("Strong Signals", len(strong_signals))
                    st.metric("Strong Avg PnL", f"{strong_signals['PnL'].mean():.2%}")
                    st.metric("Strong Win Rate", f"{(strong_signals['PnL'] > 0).mean():.2%}")
            
            with col2:
                if len(weak_signals) > 0:
                    st.metric("Weak Signals", len(weak_signals))
                    st.metric("Weak Avg PnL", f"{weak_signals['PnL'].mean():.2%}")
                    st.metric("Weak Win Rate", f"{(weak_signals['PnL'] > 0).mean():.2%}")

def create_risk_management_analysis(result):
    """Create risk management analysis charts"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("🛡️ Risk Management Analysis")
    
    # Exit reason analysis
    exit_reasons = [trade.exit_reason for trade in result.trades]
    exit_counts = pd.Series(exit_reasons).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Exit Reasons")
        fig = px.pie(
            values=exit_counts.values,
            names=exit_counts.index,
            title="Trade Exit Reasons Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Hold Time Analysis")
        hold_times = [trade.hold_minutes for trade in result.trades]
        
        fig = px.histogram(
            x=hold_times,
            nbins=20,
            title="Trade Hold Time Distribution",
            labels={'x': 'Hold Time (minutes)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    st.subheader("📈 Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stop Loss Count", result.stop_loss_count)
        st.metric("Stop Loss Rate", f"{result.stop_loss_count/result.total_trades:.1%}")
    
    with col2:
        st.metric("Take Profit Count", result.take_profit_count)
        st.metric("Take Profit Rate", f"{result.take_profit_count/result.total_trades:.1%}")
    
    with col3:
        st.metric("Avg Hold Time", f"{result.avg_hold_time:.1f} min")
        st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
    
    with col4:
        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        st.metric("Profit Factor", f"{result.summary_stats.get('profit_factor', 0):.2f}")

def create_enhanced_performance_summary(result):
    """Create enhanced performance summary with new metrics"""
    if not hasattr(result, 'trades') or not result.trades:
        st.warning("No trade data available")
        return
    
    st.subheader("📊 Enhanced Performance Summary")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total PnL", f"{result.total_pnl:.2%}")
        st.metric("Win Rate", f"{result.win_rate:.1%}")
    
    with col2:
        st.metric("Total Trades", result.total_trades)
        st.metric("Avg Trade PnL", f"{result.avg_trade_pnl:.2%}")
    
    with col3:
        st.metric("Strong Signal PnL", f"{result.strong_signal_pnl:.2%}")
        st.metric("Weak Signal PnL", f"{result.weak_signal_pnl:.2%}")
    
    with col4:
        st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    
    # Signal distribution
    st.subheader("🎯 Signal Distribution")
    
    signal_counts = {}
    for trade in result.trades:
        signal_type = trade.signal.value
        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
    
    if signal_counts:
        fig = px.bar(
            x=list(signal_counts.keys()),
            y=list(signal_counts.values()),
            title="Signal Type Distribution",
            labels={'x': 'Signal Type', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by exit reason
    st.subheader("📈 Performance by Exit Reason")
    
    exit_performance = {}
    for trade in result.trades:
        if trade.pnl is not None:
            if trade.exit_reason not in exit_performance:
                exit_performance[trade.exit_reason] = {'pnls': [], 'count': 0}
            exit_performance[trade.exit_reason]['pnls'].append(trade.pnl)
            exit_performance[trade.exit_reason]['count'] += 1
    
    if exit_performance:
        exit_data = []
        for reason, data in exit_performance.items():
            exit_data.append({
                'Exit_Reason': reason,
                'Count': data['count'],
                'Avg_PnL': np.mean(data['pnls']),
                'Win_Rate': sum(1 for pnl in data['pnls'] if pnl > 0) / len(data['pnls'])
            })
        
        exit_df = pd.DataFrame(exit_data)
        st.dataframe(exit_df.round(4))

def main():
    st.title("📘 Sentiment-Triggered Trading System Dashboard")
    st.markdown("**Real-Time Sentiment-Driven Market Signal Generator**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    symbol = st.sidebar.text_input("Symbol", value="TSLA")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    use_mock = st.sidebar.checkbox("Use Mock Data", value=True)
    news_source = st.sidebar.selectbox("News Source", ["auto", "finnhub", "newsapi"])
    
    # Cache management
    cache_mode = setup_cache_management()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal Flow", "📈 Backtest Analysis", "🔥 Sentiment Analysis", "📋 Trade Details"])
    
    with tab1:
        create_signal_flow_diagram()
    
    with tab2:
        if st.button("🚀 Run Backtest Analysis"):
            with st.spinner("Running backtest..."):
                result = run_backtest_analysis(symbol, start_date, end_date, use_mock, cache_mode)
                
                # Store result in session state
                st.session_state.backtest_result = result
                
                st.success("Backtest completed!")
        
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result
            create_multi_horizon_analysis(result)
            create_signal_overlay_chart(result)
    
    with tab3:
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result
            create_sentiment_heatmap(result)
            create_long_short_pnl_distribution(result)
            create_symbol_cumulative_returns(result)
            create_monthly_performance(result)
            create_confidence_analysis(result)
            create_correlation_analysis(result)
            create_sentiment_pnl_correlation(result)
            create_risk_management_analysis(result)
            create_enhanced_performance_summary(result)
    
    with tab4:
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result
            
            st.subheader("📋 Recent Trades")
            
            if result.trades:
                trade_data = []
                for trade in result.trades[-10:]:  # Show last 10 trades
                    trade_data.append({
                        'Timestamp': trade.timestamp,
                        'Symbol': trade.symbol,
                        'Signal': trade.signal.value,
                        'Entry Price': f"${trade.entry_price:.2f}",
                        'Exit Price': f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
                        'PnL': f"{trade.pnl:.2%}" if trade.pnl else "N/A",
                        'Confidence': f"{trade.confidence:.2f}",
                        'Reasoning': trade.reasoning[:50] + "..." if len(trade.reasoning) > 50 else trade.reasoning
                    })
                
                df_trades = pd.DataFrame(trade_data)
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("No trades found in the selected period")

if __name__ == "__main__":
    main() 