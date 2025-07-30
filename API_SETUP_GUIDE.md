# 🚀 SentimentTriggeredTradingSystem - API 설정 가이드

## 📋 필요한 API 목록

### 1. **NewsAPI** (뉴스 데이터) - 필수
- **URL**: https://newsapi.org/
- **가격**: 무료 (1,000 requests/day)
- **용도**: 실시간 뉴스 헤드라인 및 내용
- **설정**: 
  ```bash
  # .env 파일에 추가
  NEWS_API_KEY=your_news_api_key_here
  ```

### 2. **Finnhub** (금융 뉴스) - 권장
- **URL**: https://finnhub.io/
- **가격**: 무료 (60 calls/minute)
- **용도**: 기업별 뉴스, 시장 감정 데이터
- **설정**:
  ```bash
  # .env 파일에 추가
  FINNHUB_API_KEY=your_finnhub_api_key_here
  ```

### 3. **Twelve Data** (실시간 가격) - 선택
- **URL**: https://twelvedata.com/
- **가격**: 무료 (800 requests/day)
- **용도**: 1분봉 실시간 가격 데이터
- **설정**:
  ```bash
  # .env 파일에 추가
  TWELVE_DATA_API_KEY=your_twelve_data_api_key_here
  ```

## 🛠️ API 설정 단계

### Step 1: API 키 발급

#### 1. NewsAPI
1. https://newsapi.org/register 접속
2. 이메일, 비밀번호 입력
3. 무료 플랜 선택
4. API 키 복사

#### 2. Finnhub
1. https://finnhub.io/register 접속
2. 계정 생성
3. 무료 플랜 선택
4. API 키 복사

#### 3. Twelve Data (선택사항)
1. https://twelvedata.com/register 접속
2. 계정 생성
3. 무료 플랜 선택
4. API 키 복사

### Step 2: 환경 변수 설정

```bash
# 프로젝트 루트에 .env 파일 생성
echo "NEWS_API_KEY=your_news_api_key_here" > .env
echo "FINNHUB_API_KEY=your_finnhub_api_key_here" >> .env
echo "TWELVE_DATA_API_KEY=your_twelve_data_api_key_here" >> .env
```

### Step 3: API 테스트

```bash
# API 연결 테스트
python -c "
from fetchers.news_fetcher import NewsAPIFetcher
from fetchers.price_fetcher import YahooFinanceFetcher
from datetime import datetime, timedelta

# 뉴스 API 테스트
try:
    news_fetcher = NewsAPIFetcher()
    news = news_fetcher.fetch_news('TSLA', datetime.now() - timedelta(days=1), datetime.now())
    print(f'✅ NewsAPI 연결 성공: {len(news)} 개 뉴스')
except Exception as e:
    print(f'❌ NewsAPI 연결 실패: {e}')

# 가격 API 테스트
try:
    price_fetcher = YahooFinanceFetcher()
    prices = price_fetcher.fetch_prices('TSLA', '1m', datetime.now() - timedelta(hours=1), datetime.now())
    print(f'✅ Yahoo Finance 연결 성공: {len(prices)} 개 가격 데이터')
except Exception as e:
    print(f'❌ Yahoo Finance 연결 실패: {e}')
"
```

## 📊 API별 데이터 비교

### NewsAPI vs Finnhub

| 기능 | NewsAPI | Finnhub |
|------|---------|---------|
| **뉴스 소스** | 80,000+ 소스 | 금융 특화 |
| **업데이트** | 실시간 | 실시간 |
| **제한** | 1,000/day | 60/min |
| **특징** | 일반 뉴스 | 금융 뉴스 |
| **추천** | ✅ 기본 | ✅ 고급 |

### Yahoo Finance vs Twelve Data

| 기능 | Yahoo Finance | Twelve Data |
|------|---------------|-------------|
| **지연** | 15분 | 실시간 |
| **제한** | 없음 | 800/day |
| **정확도** | 높음 | 매우 높음 |
| **추천** | ✅ 기본 | ✅ 고급 |

## 🎯 권장 설정

### 기본 설정 (무료)
```bash
# .env 파일
NEWS_API_KEY=your_key_here
# Yahoo Finance는 API 키 불필요
```

### 고급 설정 (무료)
```bash
# .env 파일
NEWS_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
```

## 🚀 실제 사용 예시

### 1. 기본 설정으로 시작
```bash
# NewsAPI + Yahoo Finance
python main.py --backtest TSLA --real
```

### 2. 고급 설정으로 업그레이드
```bash
# 모든 API 사용
python main.py --backtest TSLA NVDA AAPL --real
```

### 3. 실시간 모니터링
```bash
# 실시간 데이터로 모니터링
python main.py --monitor TSLA --real --duration 60
```

## ⚠️ 주의사항

### API 제한
- **NewsAPI**: 1,000 requests/day
- **Finnhub**: 60 calls/minute
- **Twelve Data**: 800 requests/day

### 비용
- 모든 API가 **무료 플랜** 제공
- 무료 플랜으로도 충분히 테스트 가능

### 대안
- **Yahoo Finance**: API 키 불필요, 무제한 사용
- **Mock 데이터**: API 키 없이도 완전 테스트 가능

## 🎉 완료 후 확인

```bash
# 1. API 연결 테스트
python -c "from fetchers.news_fetcher import NewsAPIFetcher; print('✅ API 설정 완료')"

# 2. 실제 데이터로 백테스트
python main.py --backtest TSLA --real

# 3. 실시간 모니터링
python main.py --monitor TSLA --real --duration 10
```

## 📞 문제 해결

### API 키 오류
```bash
# .env 파일 확인
cat .env

# 환경 변수 확인
python -c "import os; print(os.getenv('NEWS_API_KEY'))"
```

### 네트워크 오류
```bash
# 인터넷 연결 확인
ping newsapi.org

# API 상태 확인
curl https://newsapi.org/v2/everything?q=TSLA&apiKey=YOUR_KEY
```

---

**🎯 결론**: NewsAPI만 설정해도 실제 뉴스 데이터로 시스템을 테스트할 수 있습니다! 