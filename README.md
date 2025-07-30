# 🎯 SentimentTriggeredTradingSystem

**실시간 감정 분석 + 시세 기반 수동 매매 프레임워크**

AI-powered trading framework that combines real-time sentiment analysis with price data to generate trading signals for manual trading assistance.

## 🚀 Features

- **Real-time Sentiment Analysis**: Analyze news sentiment using FinBERT, VADER, or custom models
- **Price Data Integration**: Fetch 1-minute candlestick data from multiple sources
- **Signal Generation**: Generate LONG/SHORT/HOLD signals based on relative sentiment and price trends
- **Backtesting Engine**: Comprehensive backtesting with performance metrics
- **Real-time Monitoring**: Live monitoring of symbols with sentiment and signal updates
- **Visualization**: Interactive charts and performance analysis
- **Modular Architecture**: OOP design for easy extension and customization

## 📦 Architecture

```
SentimentTriggeredTradingSystem/
├── fetchers/           # Data fetching components
│   ├── news_fetcher.py      # News data sources
│   └── price_fetcher.py     # Price data sources
├── analyzers/          # Analysis components
│   └── sentiment_analyzer.py # Sentiment analysis engines
├── engine/             # Core logic
│   └── signal_engine.py     # Signal generation logic
├── backtest/           # Backtesting
│   └── backtester.py        # Backtesting engine
├── visualizer.py       # Visualization tools
├── main.py            # Main execution file
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎯 Core Strategy

The system implements Zeu's sentiment-based trading strategy:

1. **상대감정 ≥ 1.0 and 하락 추세 → SHORT**
2. **상대감정 ≤ -1.0 and 상승 추세 → LONG**
3. **Otherwise → HOLD**

Where:
- **상대감정** = Symbol Sentiment - Market Sentiment
- **Price Trends** are analyzed using moving averages and price changes
- **Volume** is considered for additional confirmation

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/SentimentTriggeredTradingSystem.git
cd SentimentTriggeredTradingSystem
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys** (optional, for real data):
```bash
# Create .env file
echo "NEWS_API_KEY=your_news_api_key" > .env
echo "FINNHUB_API_KEY=your_finnhub_api_key" >> .env
echo "TWELVE_DATA_API_KEY=your_twelve_data_api_key" >> .env
```

## 🚀 Quick Start

### Demo Mode (Recommended for first run)
```bash
python main.py --demo
```

### Backtest Mode
```bash
# Backtest TSLA and NVDA for the last 7 days
python main.py --backtest TSLA NVDA

# Backtest with specific dates
python main.py --backtest TSLA --start-date 2024-01-01 --end-date 2024-01-31

# Backtest with real APIs (requires API keys)
python main.py --backtest TSLA --real
```

### Real-time Monitoring
```bash
# Monitor AAPL and MSFT for 60 minutes
python main.py --monitor AAPL MSFT

# Monitor for custom duration
python main.py --monitor TSLA --duration 120
```

## 📊 Usage Examples

### 1. Demo Mode
```bash
python main.py --demo
```
Runs a complete demo with mock data showing:
- News sentiment analysis
- Price data processing
- Signal generation
- Backtest results
- Interactive visualizations

### 2. Backtesting
```bash
# Basic backtest
python main.py --backtest TSLA NVDA AAPL

# Advanced backtest with real data
python main.py --backtest TSLA --real --start-date 2024-01-01 --end-date 2024-01-31

# Backtest without visualizations
python main.py --backtest TSLA --no-viz
```

### 3. Real-time Monitoring
```bash
# Monitor multiple symbols
python main.py --monitor TSLA NVDA AAPL MSFT

# Monitor for 2 hours
python main.py --monitor TSLA --duration 120
```

## 🔧 Configuration

### API Keys (Optional)
For real data, set up API keys in `.env` file:

```env
NEWS_API_KEY=your_news_api_key
FINNHUB_API_KEY=your_finnhub_api_key
TWELVE_DATA_API_KEY=your_twelve_data_api_key
```

### Signal Engine Parameters
Modify signal generation parameters in `main.py`:

```python
signal_engine = SignalEngine(
    sentiment_threshold=1.0,        # Sentiment threshold for signals
    trend_lookback_periods=5,       # Periods for trend analysis
    volume_threshold=1.5            # Volume threshold for confirmation
)
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Total PnL**: Overall profit/loss percentage
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of executed trades
- **Average Trade PnL**: Average profit/loss per trade
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Profit Factor**: Ratio of gross profit to gross loss

## 🎨 Visualization

The system provides interactive visualizations:

- **Price vs Sentiment**: Overlay sentiment scores on price charts
- **Backtest Results**: Comprehensive performance analysis
- **Signal Analysis**: Visual representation of trading signals
- **Performance Metrics**: Detailed statistics and charts

## 🔄 Data Sources

### News Sources
- **NewsAPI**: Real-time news articles
- **Finnhub**: Financial news and sentiment
- **Mock Data**: For testing and development

### Price Data Sources
- **Yahoo Finance**: Free historical data
- **Twelve Data**: Real-time and historical data
- **Mock Data**: For testing and development

### Sentiment Analysis
- **FinBERT**: Financial-specific BERT model
- **VADER**: Rule-based sentiment analysis
- **Mock Analyzer**: For testing and development

## 🏗️ Architecture Details

### Core Components

1. **NewsFetcher**: Abstract base class for news data fetching
   - `NewsAPIFetcher`: Real news from NewsAPI
   - `FinnhubNewsFetcher`: Financial news from Finnhub
   - `MockNewsFetcher`: Mock data for testing

2. **PriceFetcher**: Abstract base class for price data fetching
   - `YahooFinanceFetcher`: Free price data
   - `TwelveDataFetcher`: Real-time price data
   - `MockPriceFetcher`: Mock data for testing

3. **SentimentAnalyzer**: Abstract base class for sentiment analysis
   - `FinBERTSentimentAnalyzer`: AI-powered financial sentiment
   - `VADERSentimentAnalyzer`: Rule-based sentiment
   - `MockSentimentAnalyzer`: Mock analysis for testing

4. **SignalEngine**: Core signal generation logic
   - Implements Zeu's strategy
   - Analyzes relative sentiment and price trends
   - Generates confidence scores and reasoning

5. **Backtester**: Comprehensive backtesting engine
   - Historical simulation
   - Performance metrics calculation
   - Trade execution simulation

6. **Visualizer**: Interactive visualization tools
   - Plotly-based charts
   - Performance analysis
   - Signal visualization

## 🔬 Technical Specifications

### Requirements
- Python 3.8+
- PyTorch 2.1.0+
- Transformers 4.35.0+
- Pandas 2.1.3+
- Plotly 5.17.0+

### Performance
- **Mock Mode**: Instant execution for testing
- **Real Mode**: Depends on API response times
- **Backtesting**: Processes historical data efficiently
- **Monitoring**: Real-time updates every minute

## 🎓 Educational Applications

This framework is designed for:
- **Financial Engineering**: Understanding sentiment-driven trading
- **AI/ML Research**: NLP and sentiment analysis applications
- **Control Systems**: Event-driven control framework concepts
- **Data Science**: Real-time data processing and analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. It is not financial advice. Always do your own research and consider consulting with financial professionals before making investment decisions.

## 🆘 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the code examples

---

**Built with ❤️ for AI-powered trading research and education** 