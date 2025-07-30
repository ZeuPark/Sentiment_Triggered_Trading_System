import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from backtest.backtester import Trade, BacktestResult
from engine.signal_engine import SignalType

class Visualizer:
    """Visualization tools for sentiment trading analysis"""
    
    def __init__(self, style: str = 'plotly'):
        self.style = style
        if style == 'matplotlib':
            plt.style.use('seaborn-v0_8')
    
    def plot_sentiment_vs_price(self, 
                               sentiment_df: pd.DataFrame, 
                               price_df: pd.DataFrame,
                               symbol: str = "Unknown") -> None:
        """
        Overlays sentiment score and price for intuitive inspection
        
        Args:
            sentiment_df: DataFrame with 'timestamp' and 'sentiment_score' columns
            price_df: DataFrame with 'Timestamp' and 'Close' columns
            symbol: Symbol name for title
        """
        
        if self.style == 'plotly':
            self._plot_sentiment_vs_price_plotly(sentiment_df, price_df, symbol)
        else:
            self._plot_sentiment_vs_price_matplotlib(sentiment_df, price_df, symbol)
    
    def _plot_sentiment_vs_price_plotly(self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, symbol: str):
        """Plot sentiment vs price using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', f'{symbol} Sentiment'),
            row_heights=[0.7, 0.3]
        )
        
        # Add price data
        fig.add_trace(
            go.Scatter(
                x=price_df['Timestamp'],
                y=price_df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add sentiment data
        if not sentiment_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_df['timestamp'],
                    y=sentiment_df['sentiment_score'],
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='red'),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Price vs Sentiment Analysis',
            xaxis_title='Time',
            height=600,
            showlegend=True
        )
        
        fig.show()
    
    def _plot_sentiment_vs_price_matplotlib(self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, symbol: str):
        """Plot sentiment vs price using Matplotlib"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price
        ax1.plot(price_df['Timestamp'], price_df['Close'], 'b-', linewidth=1)
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} Price')
        ax1.grid(True)
        
        # Plot sentiment
        if not sentiment_df.empty:
            ax2.plot(sentiment_df['timestamp'], sentiment_df['sentiment_score'], 'r-o', markersize=4)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Sentiment Score')
        ax2.set_xlabel('Time')
        ax2.set_title(f'{symbol} Sentiment')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_backtest_results(self, result: BacktestResult) -> None:
        """Plot comprehensive backtest results"""
        
        if self.style == 'plotly':
            self._plot_backtest_results_plotly(result)
        else:
            self._plot_backtest_results_matplotlib(result)
    
    def _plot_backtest_results_plotly(self, result: BacktestResult):
        """Plot backtest results using Plotly"""
        
        if not result.trades:
            print("No trades to visualize")
            return
        
        # Prepare data
        trades_df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'signal': trade.signal.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'confidence': trade.confidence
            }
            for trade in result.trades
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative PnL', 'Trade PnL Distribution',
                'Win Rate by Signal', 'Confidence vs PnL',
                'Trade Timeline', 'Performance Metrics'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Cumulative PnL
        cumulative_pnl = np.cumsum(trades_df['pnl'].fillna(0))
        fig.add_trace(
            go.Scatter(
                x=trades_df['timestamp'],
                y=cumulative_pnl,
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='green' if cumulative_pnl.iloc[-1] >= 0 else 'red')
            ),
            row=1, col=1
        )
        
        # 2. PnL Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=20,
                name='PnL Distribution',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Win Rate by Signal
        signal_stats = trades_df.groupby('signal').agg({
            'pnl': ['count', lambda x: (x > 0).sum()]
        }).round(3)
        signal_stats.columns = ['total_trades', 'winning_trades']
        signal_stats['win_rate'] = signal_stats['winning_trades'] / signal_stats['total_trades']
        
        fig.add_trace(
            go.Bar(
                x=signal_stats.index,
                y=signal_stats['win_rate'],
                name='Win Rate',
                marker_color=['green', 'red', 'gray']
            ),
            row=2, col=1
        )
        
        # 4. Confidence vs PnL
        fig.add_trace(
            go.Scatter(
                x=trades_df['confidence'],
                y=trades_df['pnl'],
                mode='markers',
                name='Confidence vs PnL',
                marker=dict(
                    color=trades_df['pnl'],
                    colorscale='RdYlGn',
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        # 5. Trade Timeline
        colors = {'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
        for signal in trades_df['signal'].unique():
            signal_trades = trades_df[trades_df['signal'] == signal]
            fig.add_trace(
                go.Scatter(
                    x=signal_trades['timestamp'],
                    y=signal_trades['pnl'],
                    mode='markers',
                    name=f'{signal} Trades',
                    marker=dict(color=colors.get(signal, 'blue'))
                ),
                row=3, col=1
            )
        
        # 6. Performance Metrics
        metrics = [
            f"Total PnL: {result.total_pnl:.2%}",
            f"Win Rate: {result.win_rate:.2%}",
            f"Total Trades: {result.total_trades}",
            f"Avg Trade: {result.avg_trade_pnl:.2%}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}"
        ]
        
        # Add metrics as text annotation instead of table
        fig.add_annotation(
            text="<br>".join(metrics),
            xref="x6", yref="y6",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # Update layout
        fig.update_layout(
            title='Backtest Results Analysis',
            height=1000,
            showlegend=True
        )
        
        fig.show()
    
    def _plot_backtest_results_matplotlib(self, result: BacktestResult):
        """Plot backtest results using Matplotlib"""
        
        if not result.trades:
            print("No trades to visualize")
            return
        
        # Prepare data
        trades_df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'signal': trade.signal.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'confidence': trade.confidence
            }
            for trade in result.trades
        ])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Cumulative PnL
        cumulative_pnl = np.cumsum(trades_df['pnl'].fillna(0))
        axes[0, 0].plot(trades_df['timestamp'], cumulative_pnl, 'g-' if cumulative_pnl.iloc[-1] >= 0 else 'r-')
        axes[0, 0].set_title('Cumulative PnL')
        axes[0, 0].grid(True)
        
        # 2. PnL Distribution
        axes[0, 1].hist(trades_df['pnl'], bins=20, alpha=0.7, color='lightblue')
        axes[0, 1].set_title('PnL Distribution')
        axes[0, 1].grid(True)
        
        # 3. Win Rate by Signal
        signal_stats = trades_df.groupby('signal').agg({
            'pnl': ['count', lambda x: (x > 0).sum()]
        })
        signal_stats.columns = ['total_trades', 'winning_trades']
        signal_stats['win_rate'] = signal_stats['winning_trades'] / signal_stats['total_trades']
        
        colors = ['green', 'red', 'gray']
        axes[0, 2].bar(signal_stats.index, signal_stats['win_rate'], color=colors[:len(signal_stats)])
        axes[0, 2].set_title('Win Rate by Signal')
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Confidence vs PnL
        scatter = axes[1, 0].scatter(trades_df['confidence'], trades_df['pnl'], 
                                    c=trades_df['pnl'], cmap='RdYlGn')
        axes[1, 0].set_title('Confidence vs PnL')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('PnL')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 5. Trade Timeline
        colors_map = {'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
        for signal in trades_df['signal'].unique():
            signal_trades = trades_df[trades_df['signal'] == signal]
            axes[1, 1].scatter(signal_trades['timestamp'], signal_trades['pnl'], 
                              label=signal, color=colors_map.get(signal, 'blue'))
        axes[1, 1].set_title('Trade Timeline')
        axes[1, 1].legend()
        
        # 6. Performance Metrics
        metrics = [
            f"Total PnL: {result.total_pnl:.2%}",
            f"Win Rate: {result.win_rate:.2%}",
            f"Total Trades: {result.total_trades}",
            f"Avg Trade: {result.avg_trade_pnl:.2%}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}"
        ]
        
        axes[1, 2].text(0.1, 0.9, '\n'.join(metrics), transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top')
        axes[1, 2].set_title('Performance Metrics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_signal_analysis(self, signals: Dict[str, List], price_data: Dict[str, pd.DataFrame]) -> None:
        """Plot signal analysis for multiple symbols"""
        
        if self.style == 'plotly':
            self._plot_signal_analysis_plotly(signals, price_data)
        else:
            self._plot_signal_analysis_matplotlib(signals, price_data)
    
    def _plot_signal_analysis_plotly(self, signals: Dict[str, List], price_data: Dict[str, pd.DataFrame]):
        """Plot signal analysis using Plotly"""
        
        fig = make_subplots(
            rows=len(signals), cols=1,
            subplot_titles=list(signals.keys()),
            shared_xaxes=True
        )
        
        for i, (symbol, signal_list) in enumerate(signals.items()):
            if symbol not in price_data:
                continue
            
            # Plot price
            fig.add_trace(
                go.Scatter(
                    x=price_data[symbol]['Timestamp'],
                    y=price_data[symbol]['Close'],
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(color='blue')
                ),
                row=i+1, col=1
            )
            
            # Plot signals
            for signal_info in signal_list:
                timestamp = signal_info['timestamp']
                signal = signal_info['signal']
                
                # Add signal markers
                color_map = {'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
                fig.add_trace(
                    go.Scatter(
                        x=[timestamp],
                        y=[price_data[symbol][price_data[symbol]['Timestamp'] >= timestamp]['Close'].iloc[0] if len(price_data[symbol][price_data[symbol]['Timestamp'] >= timestamp]) > 0 else 0],
                        mode='markers',
                        name=f'{signal.signal.value} Signal',
                        marker=dict(
                            symbol='triangle-up' if signal.signal == SignalType.LONG else 'triangle-down' if signal.signal == SignalType.SHORT else 'circle',
                            size=10,
                            color=color_map.get(signal.signal.value, 'gray')
                        ),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Signal Analysis',
            height=300 * len(signals),
            showlegend=False
        )
        
        fig.show()
    
    def _plot_signal_analysis_matplotlib(self, signals: Dict[str, List], price_data: Dict[str, pd.DataFrame]):
        """Plot signal analysis using Matplotlib"""
        
        fig, axes = plt.subplots(len(signals), 1, figsize=(15, 5 * len(signals)))
        if len(signals) == 1:
            axes = [axes]
        
        for i, (symbol, signal_list) in enumerate(signals.items()):
            if symbol not in price_data:
                continue
            
            ax = axes[i]
            
            # Plot price
            ax.plot(price_data[symbol]['Timestamp'], price_data[symbol]['Close'], 'b-', linewidth=1)
            
            # Plot signals
            for signal_info in signal_list:
                timestamp = signal_info['timestamp']
                signal = signal_info['signal']
                
                # Find price at signal time
                price_at_signal = price_data[symbol][price_data[symbol]['Timestamp'] >= timestamp]
                if len(price_at_signal) > 0:
                    price = price_at_signal.iloc[0]['Close']
                    
                    # Plot signal marker
                    color_map = {'LONG': 'green', 'SHORT': 'red', 'HOLD': 'gray'}
                    marker_map = {'LONG': '^', 'SHORT': 'v', 'HOLD': 'o'}
                    
                    ax.scatter(timestamp, price, 
                             color=color_map.get(signal.signal.value, 'gray'),
                             marker=marker_map.get(signal.signal.value, 'o'),
                             s=100, zorder=5)
            
            ax.set_title(f'{symbol} - Price and Signals')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show() 