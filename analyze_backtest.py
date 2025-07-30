#!/usr/bin/env python3
"""
Backtest Results Analyzer
간단한 백테스트 결과 분석 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from datetime import datetime

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    # Windows에서 사용 가능한 한글 폰트 찾기
    font_list = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Dotum', 'Gulim']
    
    for font_name in font_list:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path != fm.rcParams['font.sans-serif'][0]:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                print(f"한글 폰트 설정 완료: {font_name}")
                return True
        except:
            continue
    
    # 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    return False

def analyze_backtest_results():
    """백테스트 결과 분석"""
    
    # 한글 폰트 설정
    setup_korean_font()
    
    # CSV 파일 읽기
    df = pd.read_csv('multi_horizon_results.csv')
    
    print("=" * 60)
    print("📊 백테스트 결과 분석")
    print("=" * 60)
    
    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(f"총 거래 수: {len(df)}")
    print(f"고유 시점 수: {df['timestamp'].nunique()}")
    print(f"거래 대상: {df['symbol'].unique()}")
    
    # 신호 분포
    print(f"\n🎯 신호 분포:")
    signal_counts = df['signal'].value_counts()
    for signal, count in signal_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {signal}: {count}건 ({percentage:.1f}%)")
    
    # 시간대별 성과
    print(f"\n⏰ 시간대별 성과:")
    horizon_stats = df.groupby('horizon')['pnl'].agg(['mean', 'std', 'count', 'sum'])
    for horizon in ['5min', '15min', '30min', '60min']:
        if horizon in horizon_stats.index:
            stats = horizon_stats.loc[horizon]
            print(f"  {horizon}: 평균 {stats['mean']:.4f}, 총 {stats['sum']:.4f}, 거래수 {stats['count']}")
    
    # 실제 거래만 필터링 (HOLD 제외)
    actual_trades = df[df['signal'] != 'HOLD']
    
    if len(actual_trades) > 0:
        print(f"\n💰 실제 거래 성과 (HOLD 제외):")
        print(f"실제 거래 수: {len(actual_trades)}")
        
        # 신호별 성과
        signal_performance = actual_trades.groupby('signal')['pnl'].agg(['mean', 'sum', 'count'])
        for signal in signal_performance.index:
            stats = signal_performance.loc[signal]
            win_rate = len(actual_trades[(actual_trades['signal'] == signal) & (actual_trades['pnl'] > 0)]) / stats['count']
            print(f"  {signal}: 평균 {stats['mean']:.4f}, 총 {stats['sum']:.4f}, 승률 {win_rate:.2%}")
        
        # 시간대별 실제 거래 성과
        print(f"\n⏰ 시간대별 실제 거래 성과:")
        horizon_trades = actual_trades.groupby('horizon')['pnl'].agg(['mean', 'sum', 'count'])
        for horizon in ['5min', '15min', '30min', '60min']:
            if horizon in horizon_trades.index:
                stats = horizon_trades.loc[horizon]
                win_rate = len(actual_trades[(actual_trades['horizon'] == horizon) & (actual_trades['pnl'] > 0)]) / stats['count']
                print(f"  {horizon}: 평균 {stats['mean']:.4f}, 총 {stats['sum']:.4f}, 승률 {win_rate:.2%}")
    
    # 신뢰도 분석
    print(f"\n🎯 신뢰도 분석:")
    confidence_stats = df.groupby('signal')['confidence'].agg(['mean', 'std'])
    for signal in confidence_stats.index:
        stats = confidence_stats.loc[signal]
        print(f"  {signal}: 평균 신뢰도 {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # 상대 감정 점수 분석
    print(f"\n😊 상대 감정 점수 분석:")
    sentiment_stats = df.groupby('signal')['relative_sentiment'].agg(['mean', 'std'])
    for signal in sentiment_stats.index:
        stats = sentiment_stats.loc[signal]
        print(f"  {signal}: 평균 상대감정 {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # 최고 성과 거래
    if len(actual_trades) > 0:
        print(f"\n🏆 최고 성과 거래:")
        best_trade = actual_trades.loc[actual_trades['pnl'].idxmax()]
        worst_trade = actual_trades.loc[actual_trades['pnl'].idxmin()]
        
        print(f"  최고 수익: {best_trade['pnl']:.4f} ({best_trade['signal']}, {best_trade['horizon']})")
        print(f"  최대 손실: {worst_trade['pnl']:.4f} ({worst_trade['signal']}, {worst_trade['horizon']})")
    
    # 시각화
    create_visualizations(df, actual_trades)
    
    return df, actual_trades

def create_visualizations(df, actual_trades):
    """결과 시각화"""
    
    # 1. 신호 분포
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 4, 1)
    signal_counts = df['signal'].value_counts()
    plt.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%')
    plt.title('Signal Distribution', fontsize=12)
    
    # 2. 시간대별 PnL 분포
    plt.subplot(2, 4, 2)
    if len(actual_trades) > 0:
        actual_trades.boxplot(column='pnl', by='horizon', ax=plt.gca())
        plt.title('PnL by Time Horizon', fontsize=12)
        plt.suptitle('')  # Remove default title
    
    # 3. 신호별 PnL 분포
    plt.subplot(2, 4, 3)
    if len(actual_trades) > 0:
        actual_trades.boxplot(column='pnl', by='signal', ax=plt.gca())
        plt.title('PnL by Signal Type', fontsize=12)
        plt.suptitle('')
    
    # 4. 신뢰도 vs PnL
    plt.subplot(2, 4, 4)
    if len(actual_trades) > 0:
        plt.scatter(actual_trades['confidence'], actual_trades['pnl'], alpha=0.6)
        plt.xlabel('Confidence')
        plt.ylabel('PnL')
        plt.title('Confidence vs PnL', fontsize=12)
    
    # 5. 상대감정 vs PnL
    plt.subplot(2, 4, 5)
    if len(actual_trades) > 0:
        plt.scatter(actual_trades['relative_sentiment'], actual_trades['pnl'], alpha=0.6)
        plt.xlabel('Relative Sentiment')
        plt.ylabel('PnL')
        plt.title('Sentiment vs PnL', fontsize=12)
    
    # 6. 시간대별 누적 PnL
    plt.subplot(2, 4, 6)
    if len(actual_trades) > 0:
        for horizon in ['5min', '15min', '30min', '60min']:
            horizon_data = actual_trades[actual_trades['horizon'] == horizon]
            if len(horizon_data) > 0:
                cumulative_pnl = horizon_data['pnl'].cumsum()
                plt.plot(range(len(cumulative_pnl)), cumulative_pnl, label=horizon, marker='o')
        plt.xlabel('Trade Sequence')
        plt.ylabel('Cumulative PnL')
        plt.title('Cumulative PnL by Horizon', fontsize=12)
        plt.legend()
    
    # 7. 날짜별 PnL 히트맵
    plt.subplot(2, 4, 7)
    if len(actual_trades) > 0:
        # 날짜별 데이터 준비
        actual_trades['date'] = pd.to_datetime(actual_trades['timestamp']).dt.date
        daily_pnl = actual_trades.groupby(['date', 'horizon'])['pnl'].mean().unstack(fill_value=0)
        
        if not daily_pnl.empty:
            sns.heatmap(daily_pnl, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Daily Avg PnL'})
            plt.title('Daily PnL Heatmap', fontsize=12)
            plt.xlabel('Time Horizon')
            plt.ylabel('Date')
        else:
            plt.text(0.5, 0.5, 'No daily data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Daily PnL Heatmap', fontsize=12)
    
    # 8. 월별 성과
    plt.subplot(2, 4, 8)
    if len(actual_trades) > 0:
        actual_trades['month'] = pd.to_datetime(actual_trades['timestamp']).dt.to_period('M')
        monthly_pnl = actual_trades.groupby('month')['pnl'].agg(['mean', 'sum', 'count'])
        
        if not monthly_pnl.empty:
            monthly_pnl['mean'].plot(kind='bar', color='skyblue')
            plt.title('Monthly Performance', fontsize=12)
            plt.xlabel('Month')
            plt.ylabel('Avg PnL')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No monthly data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Monthly Performance', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('backtest_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 시각화 결과가 'backtest_analysis.png'에 저장되었습니다.")
    plt.show()

def create_daily_heatmap_detailed(df):
    """상세한 날짜별 히트맵 생성"""
    if len(df) == 0:
        print("데이터가 없습니다.")
        return
    
    # 날짜별 데이터 준비
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # 실제 거래만 필터링
    actual_trades = df[df['signal'] != 'HOLD']
    
    if len(actual_trades) == 0:
        print("실제 거래 데이터가 없습니다.")
        return
    
    # 날짜별 시간대별 PnL 히트맵
    plt.figure(figsize=(15, 10))
    
    # 1. 날짜별 시간대별 PnL 히트맵
    plt.subplot(2, 2, 1)
    daily_hourly_pnl = actual_trades.groupby(['date', 'hour'])['pnl'].mean().unstack(fill_value=0)
    if not daily_hourly_pnl.empty:
        sns.heatmap(daily_hourly_pnl, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Hourly Avg PnL'})
        plt.title('Daily Hourly PnL Heatmap', fontsize=12)
        plt.xlabel('Hour of Day')
        plt.ylabel('Date')
    else:
        plt.text(0.5, 0.5, 'No hourly data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Daily Hourly PnL Heatmap', fontsize=12)
    
    # 2. 신호별 날짜별 PnL 히트맵
    plt.subplot(2, 2, 2)
    signal_daily_pnl = actual_trades.groupby(['signal', 'date'])['pnl'].mean().unstack(fill_value=0)
    if not signal_daily_pnl.empty:
        sns.heatmap(signal_daily_pnl, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Daily Avg PnL by Signal'})
        plt.title('Daily PnL by Signal Type', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Signal Type')
    else:
        plt.text(0.5, 0.5, 'No signal data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Daily PnL by Signal Type', fontsize=12)
    
    # 3. 시간대별 날짜별 PnL 히트맵
    plt.subplot(2, 2, 3)
    horizon_daily_pnl = actual_trades.groupby(['horizon', 'date'])['pnl'].mean().unstack(fill_value=0)
    if not horizon_daily_pnl.empty:
        sns.heatmap(horizon_daily_pnl, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Daily Avg PnL by Horizon'})
        plt.title('Daily PnL by Time Horizon', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Time Horizon')
    else:
        plt.text(0.5, 0.5, 'No horizon data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Daily PnL by Time Horizon', fontsize=12)
    
    # 4. 승률 히트맵
    plt.subplot(2, 2, 4)
    daily_win_rate = actual_trades.groupby('date').apply(
        lambda x: (x['pnl'] > 0).sum() / len(x) if len(x) > 0 else 0
    )
    if not daily_win_rate.empty:
        daily_win_rate_df = pd.DataFrame(daily_win_rate, columns=['win_rate'])
        sns.heatmap(daily_win_rate_df.T, annot=True, fmt='.2%', cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'Daily Win Rate'})
        plt.title('Daily Win Rate', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Win Rate')
    else:
        plt.text(0.5, 0.5, 'No win rate data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Daily Win Rate', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('daily_heatmap_detailed.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 상세 히트맵이 'daily_heatmap_detailed.png'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    df, actual_trades = analyze_backtest_results()
    
    # 상세 히트맵도 생성
    create_daily_heatmap_detailed(df) 