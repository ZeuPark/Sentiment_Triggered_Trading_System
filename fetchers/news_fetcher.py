from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
from newsapi import NewsApiClient
import finnhub
import os
from dotenv import load_dotenv

load_dotenv()

class NewsFetcher(ABC):
    """Abstract base class for news fetching"""
    
    @abstractmethod
    def fetch_news(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Returns list of news articles with:
        - timestamp
        - headline
        - related symbols
        - raw content (optional)
        """
        pass

class NewsAPIFetcher(NewsFetcher):
    """NewsAPI implementation for fetching news"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NewsAPI key is required")
        self.client = NewsApiClient(api_key=self.api_key)
    
    def fetch_news(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            # Convert datetime to ISO format
            from_date = start_time.strftime('%Y-%m-%d')
            to_date = end_time.strftime('%Y-%m-%d')
            
            # Search for news about the symbol
            articles = self.client.get_everything(
                q=symbol,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt'
            )
            
            news_list = []
            for article in articles.get('articles', []):
                news_list.append({
                    'timestamp': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    'headline': article['title'],
                    'content': article.get('description', ''),
                    'source': article['source']['name'],
                    'url': article['url'],
                    'symbols': [symbol]
                })
            
            return news_list
            
        except Exception as e:
            print(f"Error fetching news from NewsAPI: {e}")
            return []

class FinnhubNewsFetcher(NewsFetcher):
    """Finnhub implementation for fetching news"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        self.client = finnhub.Client(api_key=self.api_key)
    
    def fetch_news(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch news from Finnhub"""
        try:
            # Convert to Unix timestamp
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            # Get company news
            news = self.client.company_news(symbol, start_ts, end_ts)
            
            news_list = []
            for article in news:
                news_list.append({
                    'timestamp': datetime.fromtimestamp(article['datetime']),
                    'headline': article['headline'],
                    'content': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'symbols': [symbol]
                })
            
            return news_list
            
        except Exception as e:
            print(f"Error fetching news from Finnhub: {e}")
            return []

class MockNewsFetcher(NewsFetcher):
    """Mock implementation for testing"""
    
    def fetch_news(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Generate mock news data for testing"""
        import random
        
        # Generate mock news articles
        mock_headlines = [
            f"{symbol} announces breakthrough in AI technology",
            f"{symbol} reports strong quarterly earnings",
            f"Analysts upgrade {symbol} stock rating",
            f"{symbol} faces regulatory challenges",
            f"{symbol} expands into new markets",
            f"Market volatility affects {symbol} shares",
            f"{symbol} partners with major tech company",
            f"Investors bullish on {symbol} future prospects"
        ]
        
        news_list = []
        current_time = start_time
        
        while current_time <= end_time:
            if random.random() < 0.3:  # 30% chance of news per hour
                news_list.append({
                    'timestamp': current_time,
                    'headline': random.choice(mock_headlines),
                    'content': f"Mock content for {symbol} at {current_time}",
                    'source': 'Mock News',
                    'url': f"https://mock.com/{symbol}",
                    'symbols': [symbol]
                })
            
            current_time += timedelta(hours=1)
        
        return news_list 