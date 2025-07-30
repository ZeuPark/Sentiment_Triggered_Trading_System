from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
from twelvedata import TDClient
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

class PriceFetcher(ABC):
    """Abstract base class for price data fetching"""
    
    @abstractmethod
    def fetch_prices(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Returns DataFrame with Timestamp, Open, High, Low, Close, Volume
        """
        pass

class YahooFinanceFetcher(PriceFetcher):
    """Yahoo Finance implementation for fetching price data"""
    
    def fetch_prices(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance"""
        try:
            # Convert interval to Yahoo Finance format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '1d': '1d'
            }
            
            yf_interval = interval_map.get(interval, '1m')
            
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_time,
                end=end_time,
                interval=yf_interval
            )
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Datetime': 'Timestamp',
                'Date': 'Timestamp'
            })
            
            # Ensure we have all required columns
            required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            return df[required_columns]
            
        except Exception as e:
            print(f"Error fetching prices from Yahoo Finance: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

class TwelveDataFetcher(PriceFetcher):
    """Twelve Data implementation for fetching price data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TWELVE_DATA_API_KEY')
        if not self.api_key:
            raise ValueError("Twelve Data API key is required")
        self.client = TDClient(apikey=self.api_key)
    
    def fetch_prices(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch price data from Twelve Data"""
        try:
            # Convert datetime to string format
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Get time series data
            ts = self.client.time_series(
                symbol=symbol,
                interval=interval,
                start_date=start_str,
                end_date=end_str,
                outputsize=5000
            )
            
            # Convert to DataFrame
            df = ts.as_pandas()
            
            # Rename columns to standard format
            df = df.rename(columns={
                'datetime': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Ensure we have all required columns
            required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            return df[required_columns]
            
        except Exception as e:
            print(f"Error fetching prices from Twelve Data: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

class AlpacaMarketFetcher(PriceFetcher):
    """Alpaca Market API implementation for fetching price data"""
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required")
        
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Alpaca API: {e}")
    
    def fetch_prices(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch price data from Alpaca Market API"""
        try:
            # Convert interval to Alpaca format
            interval_map = {
                '1m': '1Min',
                '5m': '5Min',
                '15m': '15Min',
                '1h': '1Hour',
                '1d': '1Day'
            }
            
            alpaca_interval = interval_map.get(interval, '1Min')
            
            # Convert datetime to string format for Alpaca
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Fetch historical data from Alpaca
            bars = self.api.get_bars(
                symbol,
                alpaca_interval,
                start=start_str,
                end=end_str,
                adjustment='all'
            )
            
            # Convert to DataFrame
            df = bars.df
            
            if df.empty:
                print(f"No data found for {symbol} from {start_time} to {end_time}")
                return pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Rename columns to standard format
            df = df.rename(columns={
                'timestamp': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Ensure we have all required columns
            required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            return df[required_columns]
            
        except Exception as e:
            print(f"Error fetching prices from Alpaca Market API: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

class MockPriceFetcher(PriceFetcher):
    """Mock implementation for testing"""
    
    def fetch_prices(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generate mock price data for testing"""
        import random
        import numpy as np
        
        # Generate time series
        time_delta = timedelta(minutes=1) if interval == '1m' else timedelta(hours=1)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=time_delta)
        
        # Generate mock price data with some trend and volatility
        base_price = 100.0
        prices = []
        
        for i, timestamp in enumerate(timestamps):
            # Add some trend and random walk
            trend = 0.1 * np.sin(i / 100)  # Cyclical trend
            noise = random.gauss(0, 0.5)    # Random noise
            price_change = trend + noise
            
            base_price += price_change
            base_price = max(base_price, 1.0)  # Ensure positive price
            
            # Generate OHLC data
            open_price = base_price
            high_price = open_price + abs(random.gauss(0, 0.5))
            low_price = open_price - abs(random.gauss(0, 0.5))
            close_price = open_price + random.gauss(0, 0.3)
            volume = random.randint(1000, 10000)
            
            prices.append({
                'Timestamp': timestamp,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        return pd.DataFrame(prices) 