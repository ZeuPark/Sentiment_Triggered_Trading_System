# 📘 Sentiment-Triggered Trading System
## Real-Time Sentiment-Driven Market Signal Generator

**Event-driven decision system based on sentiment analysis + minute-level price data**  
*Expandable to electrical engineering-based anomaly detection and decision models*

---

## 📌 Project Overview

This project integrates real-time news sentiment analysis with 1-minute candlestick data to generate trading signals based on the relative advantage between market sentiment and individual stock sentiment.

Beyond a simple trading algorithm, this system features:

- **Event-driven real-time decision making**
- **State-response centered structural system design**
- **Electrical Engineering application extensibility**

Built as an object-oriented AI decision framework.

---

## 🧱 Core Features & Architecture

### 🔧 Core Modules

| Module | Description |
|--------|-------------|
| **NewsFetcher** | Real-time/historical news collection via NewsAPI, Finnhub |
| **SentimentAnalyzer** | FinBERT-based news sentiment analysis (score: -1.0 ~ +1.0) |
| **PriceFetcher** | 1-minute candlestick data collection from Yahoo Finance, Twelve Data |
| **SignalEngine** | Stock vs Market sentiment comparison + price trend for LONG/SHORT/HOLD decisions |
| **Backtester** | Post-news return calculation → strategy performance analysis |
| **Visualizer** | Sentiment score and price visualization (heatmaps, signal charts) |

### 🧠 Core Strategy Logic

```python
relative_sentiment = symbol_sentiment - market_sentiment

if relative_sentiment ≥ 1.0 and price_trend == "DOWNTREND":
    signal = STRONG SHORT
elif relative_sentiment ≤ -1.0 and price_trend == "UPTREND":
    signal = STRONG LONG
elif relative_sentiment ≥ 0.5 and price_trend in ["DOWNTREND", "NEUTRAL"]:
    signal = WEAK SHORT
elif relative_sentiment ≤ -0.5 and price_trend in ["UPTREND", "NEUTRAL"]:
    signal = WEAK LONG
else:
    signal = HOLD
```

---

## 📊 Backtest Results Example

**Sentiment-Price Synchronization Criteria**: Track returns 5/15/30/60 minutes after news events

- **Condition-based entry** → Cumulative return/PnL calculation
- **SignalEngine decision logic** validation against actual returns
- **Multi-horizon performance analysis**

### Recent Performance (FinBERT + Real Data)
```
Total PnL: +0.76%
Win Rate: 11.76%
Total Trades: 17
Sharpe Ratio: +0.34
```

---

## 🎯 Electrical Engineering Application Examples

This system can be extended to the following electrical engineering applications:

| EE Domain Element | Software Counterpart |
|-------------------|---------------------|
| **Sensor Signals** (voltage/current/temperature) | 1-minute candlestick data |
| **External Events** (faults/weather/policy) | News sentiment information |
| **Anomaly Detection / State Judgment** | Sentiment-price fusion decision engine |
| **Autonomous Control / Trigger System** | SignalEngine → control trigger decisions |

### Potential EE Applications:
- **Power Grid Anomaly Detection**: Voltage/current patterns → fault prediction
- **Industrial IoT Monitoring**: Sensor fusion → predictive maintenance
- **Smart City Infrastructure**: Multi-source event correlation
- **Autonomous Vehicle Systems**: Environmental data → decision triggers

### 🧠 Core Strategy Logic

```python
relative_sentiment = symbol_sentiment - market_sentiment

if relative_sentiment ≥ 1.0 and price_trend == "DOWNTREND":
    signal = STRONG SHORT
elif relative_sentiment ≤ -1.0 and price_trend == "UPTREND":
    signal = STRONG LONG
elif relative_sentiment ≥ 0.5 and price_trend in ["DOWNTREND", "NEUTRAL"]:
    signal = WEAK SHORT
elif relative_sentiment ≤ -0.5 and price_trend in ["UPTREND", "NEUTRAL"]:
    signal = WEAK LONG
else:
    signal = HOLD
```

### 📊 Backtest Results Example

**Sentiment-Price Synchronization Criteria**: Track returns 5/15/30/60 minutes after news events

- **Condition-based entry** → Cumulative return/PnL calculation
- **SignalEngine decision logic** validation against actual returns
- **Multi-horizon performance analysis**

### 🎯 EE-Based Positioning Examples

This system can be extended to the following electrical engineering application systems:

| EE Domain Element | Software Counterpart |
|-------------------|---------------------|
| **Sensor Signals** (voltage/current/temperature) | 1-minute candlestick data |
| **External Events** (faults/weather/policy) | News sentiment information |
| **Anomaly Detection / State Judgment** | Sentiment-price fusion decision engine |
| **Autonomous Control / Trigger System** | SignalEngine → control trigger decisions |

---

## 🧰 Technology Stack

- **Python** (OOP-based design)
- **HuggingFace Transformers** (FinBERT)
- **Yahoo Finance, Finnhub API**
- **Pandas / Matplotlib / Plotly**
- **CLI interface-based module execution**
- **Streamlit** (optional) – visualization UI

---

## 🗂️ Folder Structure

```
SentimentTriggeredTradingSystem/
├── fetchers/           # Data collection modules
│   ├── news_fetcher.py
│   └── price_fetcher.py
├── analyzers/          # Analysis modules
│   └── sentiment_analyzer.py
├── engine/             # Decision engine
│   └── signal_engine.py
├── backtest/           # Backtesting framework
│   └── backtester.py
├── visualizer.py       # Visualization tools
├── main.py            # CLI interface
├── data/              # Data storage
├── api                # API key management
└── requirements.txt   # Dependencies
```

---

## 🚀 Usage Examples

### CLI Interface

```bash
# Demo mode with sample data
python main.py --demo

# Backtest with real APIs
python main.py --backtest TSLA --real --news-source finnhub

# Real-time monitoring
python main.py --monitor TSLA NVDA --real --duration 30

# Multi-symbol backtest
python main.py --backtest TSLA NVDA AAPL --real

# Custom date range
python main.py --backtest TSLA --start-date 2024-01-01 --end-date 2024-01-31 --real
```

### API Configuration

```bash
# Set up API keys
echo "NEWS_API_KEY=your_news_api_key" > .env
echo "FINNHUB_API_KEY=your_finnhub_key" >> .env

# Test API connections
python -c "from fetchers.news_fetcher import NewsAPIFetcher; print('✅ Connected')"
```

---

## 📈 Key Features

### 🔍 **Real-Time Monitoring**
- Live sentiment analysis of news events
- Instant signal generation and alerts
- Multi-symbol simultaneous tracking

### 📊 **Advanced Analytics**
- FinBERT financial-specific sentiment analysis
- Relative sentiment divergence modeling
- Multi-horizon return calculation (5/15/30/60 min)

### 🎯 **Flexible Decision Engine**
- Configurable sentiment thresholds
- Volume-weighted confidence scoring
- Trend alignment validation

### 📈 **Comprehensive Visualization**
- Sentiment heatmaps across symbols and time
- Market vs individual symbol signal comparison
- Performance metrics and trade analysis

---

## 🔬 Research Applications

### **Financial NLP Research**
- FinBERT model evaluation on real market data
- Sentiment-price correlation analysis
- Event-driven return prediction

### **Decision System Design**
- State-response modeling
- Event-driven architecture patterns
- Multi-source data fusion

### **Electrical Engineering Extension**
- Sensor fusion algorithms
- Anomaly detection frameworks
- Autonomous control systems

---

## 💡 Future Development Directions

### **Web-Based Monitoring Dashboard**
- Streamlit-based real-time signal monitoring
- Interactive portfolio analysis
- Custom alert system

### **Advanced Analytics**
- Portfolio-based multi-symbol analysis
- Reinforcement learning integration
- Advanced risk management

### **EE Application Versions**
- Power system anomaly detection
- Industrial IoT monitoring
- Smart infrastructure management

### **AI Enhancement**
- Reinforcement learning-based auto-trading
- Deep learning signal optimization
- Multi-modal data fusion

---

## 📚 References

- **FinBERT** (Yiyang et al.) - Financial NLP
- **Twelve Data API** - Market data
- **Finnhub API** - Financial news
- **HuggingFace Transformers** - NLP models
- **Yahoo Finance** - Price data

---

## 🤝 Contributing

This project welcomes contributions in:
- **Algorithm improvements**
- **New data sources**
- **Visualization enhancements**
- **EE application extensions**

### Development Setup
```bash
git clone https://github.com/your-username/SentimentTriggeredTradingSystem.git
cd SentimentTriggeredTradingSystem
pip install -r requirements.txt
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## ⚠️ Disclaimer

This system is designed for **research and educational purposes**. It is not financial advice. Always conduct thorough testing before any real trading applications.

---

## 🎯 Project Positioning

This is **not just a trading bot** - it's an advanced **decision-making system** that demonstrates:

- **Event-driven architecture design**
- **Multi-source data fusion**
- **Real-time decision frameworks**
- **Electrical engineering applications**

Perfect for researchers, developers, and engineers interested in:
- **Financial NLP**
- **Decision system design**
- **Sensor fusion algorithms**
- **Autonomous control systems**

---

*Built with ❤️ for advanced decision-making research* 