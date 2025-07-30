#!/usr/bin/env python3
"""
Extended Backtest Analysis
2023년 6월 ~ 2024년 6월 멀티심볼 백테스트 및 상관관계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_backtest, setup_components
from visualizer import Visualizer

def run_extended_backtest():
    """확장된 백테스트 실행"""
    
    print("🚀 Extended Backtest Analysis (2023-06 ~ 2024-06)")
    print("=" * 60)
    
    # 설정
    symbols = ["TSLA", "NVDA"]  # TSLA와 NVDA만 테스트
    # 최근 7일로 설정 (API 제한 고려)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"📊 Symbols: {symbols}")
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # 컴포넌트 설정
    news_fetcher, price_fetcher, sentiment_analyzer, signal_engine = setup_components(
        use_mock=False,  # 실제 데이터 사용
        news_source="auto"
    )
    
    # 각 심볼별 백테스트 실행
    all_results = {}
    
    for symbol in symbols:
        print(f"\n📈 Running backtest for {symbol}...")
        
        try:
            # 백테스트 실행
            result = run_backtest(
                symbols=[symbol],
                start_time=start_date,
                end_time=end_date,
                use_mock=False,  # 실제 데이터 사용
                visualize=False,
                news_source="auto",
                use_cache=True
            )
            
            all_results[symbol] = result
            print(f"✅ {symbol} backtest completed")
            
        except Exception as e:
            print(f"❌ {symbol} backtest failed: {e}")
            continue
    
    return all_results

def analyze_correlation(all_results):
    """감정 점수와 수익률 상관관계 분석"""
    
    print("\n📊 Correlation Analysis")
    print("=" * 40)
    
    correlation_data = []
    
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades:
                if hasattr(trade, 'metadata') and trade.metadata:
                    relative_sentiment = trade.metadata.get('relative_sentiment', 0)
                    correlation_data.append({
                        'symbol': symbol,
                        'relative_sentiment': relative_sentiment,
                        'pnl': trade.pnl if trade.pnl else 0,
                        'confidence': trade.confidence
                    })
    
    if not correlation_data:
        print("❌ No correlation data available")
        return None
    
    df = pd.DataFrame(correlation_data)
    
    # 전체 상관관계
    correlation = df['relative_sentiment'].corr(df['pnl'])
    print(f"📈 Overall Correlation (Sentiment vs PnL): {correlation:.4f}")
    
    # 심볼별 상관관계
    print("\n📊 Symbol-wise Correlation:")
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        corr = symbol_data['relative_sentiment'].corr(symbol_data['pnl'])
        print(f"  {symbol}: {corr:.4f}")
    
    # 신뢰도별 상관관계
    print("\n🎯 Confidence-based Correlation:")
    high_conf = df[df['confidence'] >= 0.7]
    low_conf = df[df['confidence'] < 0.7]
    
    if len(high_conf) > 0:
        high_corr = high_conf['relative_sentiment'].corr(high_conf['pnl'])
        print(f"  High Confidence (≥0.7): {high_corr:.4f}")
    
    if len(low_conf) > 0:
        low_corr = low_conf['relative_sentiment'].corr(low_conf['pnl'])
        print(f"  Low Confidence (<0.7): {low_corr:.4f}")
    
    return df

def create_extended_visualizations(all_results, correlation_df):
    """확장된 시각화 생성"""
    
    print("\n📊 Creating Extended Visualizations...")
    
    # 1. 종목별 누적 수익률
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            trades_df = pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'pnl': trade.pnl if trade.pnl else 0
                }
                for trade in result.trades
            ])
            if len(trades_df) > 0:
                cumulative_pnl = trades_df['pnl'].cumsum()
                plt.plot(range(len(cumulative_pnl)), cumulative_pnl, label=symbol, marker='o', markersize=2)
    
    plt.title('Cumulative PnL by Symbol')
    plt.xlabel('Trade Sequence')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.grid(True)
    
    # 2. 롱/숏별 PnL 분포
    plt.subplot(2, 3, 2)
    long_pnls = []
    short_pnls = []
    
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades:
                if trade.pnl:
                    if trade.signal.value == 'LONG':
                        long_pnls.append(trade.pnl)
                    elif trade.signal.value == 'SHORT':
                        short_pnls.append(trade.pnl)
    
    if long_pnls:
        plt.hist(long_pnls, alpha=0.7, label='LONG', bins=20, color='green')
    if short_pnls:
        plt.hist(short_pnls, alpha=0.7, label='SHORT', bins=20, color='red')
    
    plt.title('PnL Distribution by Signal Type')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. 월별 수익률
    plt.subplot(2, 3, 3)
    monthly_data = []
    
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades:
                if trade.pnl:
                    monthly_data.append({
                        'symbol': symbol,
                        'month': trade.timestamp.strftime('%Y-%m'),
                        'pnl': trade.pnl
                    })
    
    if monthly_data:
        monthly_df = pd.DataFrame(monthly_data)
        monthly_pnl = monthly_df.groupby('month')['pnl'].mean()
        monthly_pnl.plot(kind='bar', color='skyblue')
        plt.title('Monthly Average PnL')
        plt.xlabel('Month')
        plt.ylabel('Average PnL')
        plt.xticks(rotation=45)
    
    # 4. 감정 점수 vs PnL 상관관계
    plt.subplot(2, 3, 4)
    if correlation_df is not None and len(correlation_df) > 0:
        plt.scatter(correlation_df['relative_sentiment'], correlation_df['pnl'], alpha=0.6)
        plt.xlabel('Relative Sentiment')
        plt.ylabel('PnL')
        plt.title('Sentiment vs PnL Correlation')
        
        # 상관관계 선 추가
        z = np.polyfit(correlation_df['relative_sentiment'], correlation_df['pnl'], 1)
        p = np.poly1d(z)
        plt.plot(correlation_df['relative_sentiment'], p(correlation_df['relative_sentiment']), "r--", alpha=0.8)
    
    # 5. 신뢰도별 성과 비교
    plt.subplot(2, 3, 5)
    if correlation_df is not None and len(correlation_df) > 0:
        high_conf_pnl = correlation_df[correlation_df['confidence'] >= 0.7]['pnl']
        low_conf_pnl = correlation_df[correlation_df['confidence'] < 0.7]['pnl']
        
        if len(high_conf_pnl) > 0 and len(low_conf_pnl) > 0:
            plt.boxplot([high_conf_pnl, low_conf_pnl], labels=['High Conf (≥0.7)', 'Low Conf (<0.7)'])
            plt.title('Performance by Confidence Level')
            plt.ylabel('PnL')
    
    # 6. 종목별 성과 요약
    plt.subplot(2, 3, 6)
    performance_summary = []
    
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            total_pnl = result.total_pnl
            win_rate = result.win_rate
            performance_summary.append({
                'Symbol': symbol,
                'Total PnL': total_pnl,
                'Win Rate': win_rate
            })
    
    if performance_summary:
        perf_df = pd.DataFrame(performance_summary)
        x = range(len(perf_df))
        plt.bar(x, perf_df['Total PnL'], color=['green' if p > 0 else 'red' for p in perf_df['Total PnL']])
        plt.title('Total PnL by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Total PnL')
        plt.xticks(x, perf_df['Symbol'])
    
    plt.tight_layout()
    plt.savefig('extended_backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Extended analysis saved as 'extended_backtest_analysis.png'")
    plt.show()

def print_performance_summary(all_results):
    """성과 요약 출력"""
    
    print("\n📊 Performance Summary")
    print("=" * 60)
    
    summary_data = []
    
    for symbol, result in all_results.items():
        if hasattr(result, 'trades') and result.trades:
            summary_data.append({
                'Symbol': symbol,
                'Total PnL': f"{result.total_pnl:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Total Trades': result.total_trades,
                'Avg Trade PnL': f"{result.avg_trade_pnl:.2%}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # 전체 포트폴리오 성과 계산
        total_pnl = sum([float(s['Total PnL'].rstrip('%'))/100 for s in summary_data])
        avg_win_rate = sum([float(s['Win Rate'].rstrip('%'))/100 for s in summary_data]) / len(summary_data)
        total_trades = sum([s['Total Trades'] for s in summary_data])
        
        print(f"\n🏆 Portfolio Summary:")
        print(f"  Average Total PnL: {total_pnl/len(summary_data):.2%}")
        print(f"  Average Win Rate: {avg_win_rate:.2%}")
        print(f"  Total Trades: {total_trades}")

def main():
    """메인 실행 함수"""
    
    # 확장된 백테스트 실행
    all_results = run_extended_backtest()
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # 상관관계 분석
    correlation_df = analyze_correlation(all_results)
    
    # 성과 요약 출력
    print_performance_summary(all_results)
    
    # 확장된 시각화 생성
    create_extended_visualizations(all_results, correlation_df)
    
    print("\n🎉 Extended backtest analysis completed!")

if __name__ == "__main__":
    main() 