#!/usr/bin/env python3
"""
Complete Sentiment Trading System Analysis
전체 시스템을 한 번에 실행하는 통합 스크립트
"""

import os
import sys
import subprocess
from datetime import datetime
import time

def print_banner():
    """배너 출력"""
    print("🎯" + "="*60)
    print("🚀 Sentiment Trading System - Complete Analysis")
    print("="*60)
    print("📊 Comprehensive backtest and analysis pipeline")
    print("📅 Period: 2023-06-01 to 2024-06-30")
    print("🎯 Symbols: TSLA, NVDA, AAPL, AMZN")
    print("="*60)

def check_environment():
    """환경 체크"""
    print("🔍 Checking environment...")
    
    # 가상환경 활성화 확인
    if not os.path.exists('.venv'):
        print("❌ Virtual environment not found. Please activate .venv")
        return False
    
    # 필요한 파일들 확인
    required_files = [
        'main.py',
        'extended_backtest.py',
        'dashboard.py',
        'engine/signal_engine.py',
        'backtest/backtester.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return False
    
    print("✅ Environment check passed")
    return True

def run_step(step_name, command, description=""):
    """단계별 실행"""
    print(f"\n🔄 {step_name}")
    if description:
        print(f"   {description}")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {step_name} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {step_name} failed")
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {step_name} failed with error: {e}")
        return False
    
    end_time = time.time()
    print(f"⏱️  Duration: {end_time - start_time:.2f} seconds")
    
    return True

def main():
    """메인 실행 함수"""
    
    print_banner()
    
    # 환경 체크
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        return
    
    print("\n🚀 Starting complete analysis pipeline...")
    
    # 1. 기본 백테스트 (TSLA)
    print("\n" + "="*60)
    print("📊 STEP 1: Basic Backtest (TSLA)")
    print("="*60)
    
    success = run_step(
        "Basic TSLA Backtest",
        "python main.py --backtest TSLA --start-date 2024-01-01 --end-date 2024-01-31 --real",
        "Running basic backtest for TSLA with real data"
    )
    
    if not success:
        print("❌ Basic backtest failed. Stopping pipeline.")
        return
    
    # 2. 확장된 백테스트 (멀티심볼)
    print("\n" + "="*60)
    print("📊 STEP 2: Extended Multi-Symbol Backtest")
    print("="*60)
    
    success = run_step(
        "Extended Backtest",
        "python extended_backtest.py",
        "Running extended backtest for TSLA, NVDA, AAPL, AMZN"
    )
    
    if not success:
        print("⚠️  Extended backtest failed. Continuing with available data.")
    
    # 3. 결과 분석
    print("\n" + "="*60)
    print("📊 STEP 3: Result Analysis")
    print("="*60)
    
    success = run_step(
        "Backtest Analysis",
        "python analyze_backtest.py",
        "Analyzing backtest results and generating visualizations"
    )
    
    if not success:
        print("⚠️  Analysis failed. Continuing...")
    
    # 4. Streamlit 대시보드 시작
    print("\n" + "="*60)
    print("📊 STEP 4: Streamlit Dashboard")
    print("="*60)
    
    print("🌐 Starting Streamlit dashboard...")
    print("   Dashboard will be available at: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard")
    print("\n🚀 Starting dashboard in background...")
    
    try:
        # 백그라운드에서 Streamlit 시작
        dashboard_process = subprocess.Popen(
            "streamlit run dashboard.py --server.port 8501",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("✅ Dashboard started successfully")
        print("📊 You can now access the dashboard at: http://localhost:8501")
        print("⏹️  To stop the dashboard, press Ctrl+C")
        
        # 대시보드가 실행되는 동안 대기
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping dashboard...")
            dashboard_process.terminate()
            dashboard_process.wait()
            print("✅ Dashboard stopped")
            
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
    
    # 5. 최종 요약
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETED")
    print("="*60)
    
    print("📊 Generated Files:")
    generated_files = [
        'backtest_analysis.png',
        'daily_heatmap_detailed.png',
        'extended_backtest_analysis.png',
        'multi_horizon_results.csv'
    ]
    
    for file in generated_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (not found)")
    
    print("\n📈 Next Steps:")
    print("   1. Review the generated visualizations")
    print("   2. Check the Streamlit dashboard for interactive analysis")
    print("   3. Analyze correlation results")
    print("   4. Optimize strategy parameters based on findings")
    
    print("\n🎯 Quick Commands:")
    print("   • Run basic backtest: python main.py --backtest TSLA")
    print("   • Run extended analysis: python extended_backtest.py")
    print("   • Start dashboard: streamlit run dashboard.py")
    print("   • Analyze results: python analyze_backtest.py")

if __name__ == "__main__":
    main() 