#!/usr/bin/env python3
"""
Parameter Optimization Script for Grid Search
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product
from dotenv import load_dotenv
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from fetchers.news_fetcher import NewsAPIFetcher
from fetchers.price_fetcher import AlpacaMarketFetcher
from analyzers.sentiment_analyzer import FinBERTSentimentAnalyzer
from engine.signal_engine import SignalEngine
from backtest.backtester import Backtester

class ParameterOptimizer:
    """Parameter optimization for trading strategy"""
    
    def __init__(self):
        """Initialize the optimizer with data fetchers"""
        self.news_fetcher = NewsAPIFetcher()
        self.price_fetcher = AlpacaMarketFetcher()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
    
    def run_grid_search(self, 
                       symbols: List[str],
                       start_time: datetime,
                       end_time: datetime,
                       param_grid: Dict = None) -> pd.DataFrame:
        """
        Run grid search over parameter combinations
        
        Args:
            symbols: List of symbols to test
            start_time: Start of test period
            end_time: End of test period
            param_grid: Dictionary of parameter ranges to test
            
        Returns:
            DataFrame with results for each parameter combination
        """
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'sentiment_threshold': [0.2, 0.3, 0.4, 0.5],
                'news_confidence_threshold': [0.4, 0.5, 0.6, 0.7],
                'strong_signal_threshold': [0.6, 0.7, 0.8, 0.9],
                'stop_loss_pct': [0.02, 0.03, 0.04],
                'take_profit_pct': [0.04, 0.05, 0.06]
            }
        
        print(f"🔍 Starting grid search with {self._count_combinations(param_grid)} combinations")
        print(f"📅 Test period: {start_time} to {end_time}")
        print(f"📊 Symbols: {symbols}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        # Fetch data once for all tests
        print("📊 Fetching data...")
        news_data = self._fetch_data_for_symbols(symbols, start_time, end_time)
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            print(f"\n🧪 Testing combination {i+1}/{len(combinations)}: {params}")
            
            try:
                # Create signal engine with current parameters
                signal_engine = SignalEngine(
                    sentiment_threshold=params.get('sentiment_threshold', 0.3),
                    news_confidence_threshold=params.get('news_confidence_threshold', 0.6),
                    strong_signal_threshold=params.get('strong_signal_threshold', 0.8),
                    price_momentum_weight=0.3
                )
                
                # Create backtester with current parameters
                backtester = Backtester(
                    news_fetcher=self.news_fetcher,
                    price_fetcher=self.price_fetcher,
                    sentiment_analyzer=self.sentiment_analyzer,
                    signal_engine=signal_engine,
                    stop_loss_pct=params.get('stop_loss_pct', 0.03),
                    take_profit_pct=params.get('take_profit_pct', 0.05)
                )
                
                # Run backtest
                result = backtester.run(symbols, start_time, end_time, '1m')
                
                # Store results
                result_dict = {
                    **params,
                    'total_pnl': result.total_pnl,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'avg_trade_pnl': result.avg_trade_pnl,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'strong_signal_pnl': result.strong_signal_pnl,
                    'weak_signal_pnl': result.weak_signal_pnl,
                    'stop_loss_count': result.stop_loss_count,
                    'take_profit_count': result.take_profit_count,
                    'avg_hold_time': result.avg_hold_time,
                    'profit_factor': result.summary_stats.get('profit_factor', 0),
                    'strong_signals': result.summary_stats.get('strong_signals', 0),
                    'weak_signals': result.summary_stats.get('weak_signals', 0)
                }
                
                results.append(result_dict)
                
                print(f"✅ Result: PnL={result.total_pnl:.2%}, Win Rate={result.win_rate:.1%}, Trades={result.total_trades}")
                
            except Exception as e:
                print(f"❌ Error with combination {combination}: {e}")
                continue
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by different metrics
        print(f"\n📊 Grid search completed! Found {len(df_results)} valid combinations")
        
        return df_results
    
    def _fetch_data_for_symbols(self, symbols: List[str], start_time: datetime, end_time: datetime) -> Dict:
        """Fetch and cache data for all symbols"""
        data = {}
        
        for symbol in symbols:
            print(f"📈 Fetching data for {symbol}...")
            try:
                news = self.news_fetcher.fetch_news(symbol, start_time, end_time)
                prices = self.price_fetcher.fetch_prices(symbol, '1m', start_time, end_time)
                
                data[symbol] = {
                    'news': news,
                    'prices': prices
                }
                
                print(f"✅ {symbol}: {len(news)} news, {len(prices)} prices")
                
            except Exception as e:
                print(f"❌ Error fetching data for {symbol}: {e}")
                data[symbol] = {'news': [], 'prices': pd.DataFrame()}
        
        return data
    
    def _count_combinations(self, param_grid: Dict) -> int:
        """Count total number of parameter combinations"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def analyze_results(self, df_results: pd.DataFrame) -> Dict:
        """Analyze optimization results and provide recommendations"""
        
        if df_results.empty:
            return {}
        
        analysis = {}
        
        # Best combinations by different metrics
        metrics = ['total_pnl', 'sharpe_ratio', 'win_rate', 'profit_factor']
        
        for metric in metrics:
            if metric in df_results.columns:
                best_idx = df_results[metric].idxmax()
                best_params = df_results.loc[best_idx]
                
                analysis[f'best_{metric}'] = {
                    'params': {col: best_params[col] for col in df_results.columns if col not in metrics},
                    'performance': {metric: best_params[metric] for metric in metrics if metric in df_results.columns}
                }
        
        # Parameter sensitivity analysis
        param_sensitivity = {}
        param_columns = [col for col in df_results.columns if col not in metrics + ['total_trades', 'strong_signals', 'weak_signals', 'stop_loss_count', 'take_profit_count', 'avg_hold_time']]
        
        for param in param_columns:
            if param in df_results.columns:
                # Calculate correlation with performance metrics
                correlations = {}
                for metric in metrics:
                    if metric in df_results.columns:
                        corr = df_results[param].corr(df_results[metric])
                        correlations[metric] = corr
                
                param_sensitivity[param] = correlations
        
        analysis['parameter_sensitivity'] = param_sensitivity
        
        # Summary statistics
        analysis['summary'] = {
            'total_combinations': len(df_results),
            'avg_pnl': df_results['total_pnl'].mean(),
            'best_pnl': df_results['total_pnl'].max(),
            'worst_pnl': df_results['total_pnl'].min(),
            'avg_sharpe': df_results['sharpe_ratio'].mean(),
            'best_sharpe': df_results['sharpe_ratio'].max(),
            'profitable_combinations': (df_results['total_pnl'] > 0).sum(),
            'profitable_rate': (df_results['total_pnl'] > 0).mean()
        }
        
        return analysis
    
    def save_results(self, df_results: pd.DataFrame, analysis: Dict, filename: str = None):
        """Save results to CSV and analysis to JSON"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parameter_optimization_{timestamp}"
        
        # Save results DataFrame
        df_results.to_csv(f"{filename}_results.csv", index=False)
        print(f"💾 Results saved to {filename}_results.csv")
        
        # Save analysis
        import json
        with open(f"{filename}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"💾 Analysis saved to {filename}_analysis.json")
        
        return filename

def main():
    """Main function to run parameter optimization"""
    
    print("🚀 Parameter Optimization for Sentiment Trading Strategy")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Test parameters
    symbols = ['TSLA', 'NVDA']
    start_time = datetime(2025, 7, 1, 9, 30)
    end_time = datetime(2025, 7, 3, 16, 0)
    
    # Define parameter grid (smaller for faster testing)
    param_grid = {
        'sentiment_threshold': [0.2, 0.3, 0.4],
        'news_confidence_threshold': [0.5, 0.6, 0.7],
        'strong_signal_threshold': [0.7, 0.8, 0.9],
        'stop_loss_pct': [0.02, 0.03, 0.04],
        'take_profit_pct': [0.04, 0.05, 0.06]
    }
    
    # Run grid search
    results = optimizer.run_grid_search(symbols, start_time, end_time, param_grid)
    
    if not results.empty:
        # Analyze results
        analysis = optimizer.analyze_results(results)
        
        # Display best results
        print("\n🏆 Best Parameter Combinations:")
        print("=" * 40)
        
        metrics = ['total_pnl', 'sharpe_ratio', 'win_rate', 'profit_factor']
        for metric in metrics:
            if f'best_{metric}' in analysis:
                best = analysis[f'best_{metric}']
                print(f"\n📈 Best {metric.upper()}:")
                print(f"   Parameters: {best['params']}")
                print(f"   Performance: {best['performance']}")
        
        # Display parameter sensitivity
        print("\n📊 Parameter Sensitivity Analysis:")
        print("=" * 40)
        
        sensitivity = analysis.get('parameter_sensitivity', {})
        for param, correlations in sensitivity.items():
            print(f"\n🔧 {param}:")
            for metric, corr in correlations.items():
                print(f"   {metric}: {corr:.3f}")
        
        # Display summary
        summary = analysis.get('summary', {})
        print(f"\n📋 Summary:")
        print(f"   Total combinations tested: {summary.get('total_combinations', 0)}")
        print(f"   Profitable combinations: {summary.get('profitable_combinations', 0)} ({summary.get('profitable_rate', 0):.1%})")
        print(f"   Best PnL: {summary.get('best_pnl', 0):.2%}")
        print(f"   Best Sharpe: {summary.get('best_sharpe', 0):.2f}")
        
        # Save results
        filename = optimizer.save_results(results, analysis)
        
        # Display top 10 results
        print(f"\n🏅 Top 10 Results by PnL:")
        print("=" * 40)
        top_results = results.nlargest(10, 'total_pnl')
        print(top_results[['sentiment_threshold', 'news_confidence_threshold', 'strong_signal_threshold', 'stop_loss_pct', 'take_profit_pct', 'total_pnl', 'sharpe_ratio', 'win_rate']].round(4))
        
    else:
        print("❌ No valid results found")

if __name__ == "__main__":
    main() 