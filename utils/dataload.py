import streamlit as st 
import pandas as pd
import json
import yfinance as yf
import utils.functions as fn
import os

BASE_DIR = os.path.dirname(__file__)  # folder where dataload.py is
file_path = os.path.join(BASE_DIR, "data", "stocks_data.json")

with open(file_path, "r", encoding="utf-8") as f:
    stocks_data_json = json.load(f)
    
@st.cache_data
def get_display_to_symbol(region, market):
    try:
        stock_entries = stocks_data_json["market"].get(market, [])
        return {
            f"{entry['Symbol']} – {entry.get('Security', 'Unknown')}": entry['Symbol']
            for entry in stock_entries
        }
    except Exception as e:
        st.warning(f"⚠ Could not build display mapping for {market}: {str(e)}")
        return {
            symbol: symbol
            for symbol in region_market_stocks[region][market]
        }
        
@st.cache_data
def get_display_to_symbol_market(market: str) -> dict:
    try:
        stock_entries = stocks_data_json.get("market", {}).get(market, [])
        if not stock_entries:
            st.warning(f"⚠ No stock entries found for market: {market}")
            return {}

        return {
            f"{entry['Symbol']} – {entry.get('Security', 'Unknown')}": entry['Symbol']
            for entry in stock_entries
        }

    except Exception as e:
        st.warning(f"⚠ Error building display mapping for {market}: {str(e)}")
        return {}
    
@st.cache_data
def get_ticker_from_display(selected_display: str, market: str) -> str | None:
    display_mapping = get_display_to_symbol_market(market)
    return display_mapping.get(selected_display)

# Define region → market → default stocks
region_market_stocks = {
        'Asia': {
        'Nifty 50': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'INFY.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
            'BHARTIARTL.NS', 'BAJFINANCE.NS'],
        'Nifty 500': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
        'Nikkei 225': ['7203.T', '9984.T'],
        'Hang Seng': ['0005.HK', '0700.HK'],

        },
    'North America': {
        'S&P 500':['AAPL', 'TSLA', 'AMZN', 'GOOGL'] ,
        'NASDAQ 100': ['NVDA', 'NFLX', 'MSFT', 'META'],
        'Dow Jones': ['MCD', 'IBM', 'JNJ', 'DIS']
    },
    'Europe': {
        'FTSE 100': ['HSBA.L', 'BP.L'],
        'DAX': ['SIE.DE', 'BMW.DE'],
        'CAC 40': ['AIR.PA', 'OR.PA']
    },

}

# Market → Yahoo Finance ticker
market_tickers = {
    'S&P 500': '^GSPC',
    'NASDAQ 100': '^NDX',
    'Dow Jones': '^DJI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'CAC 40': '^FCHI',
    'Nifty 50': '^NSEI',
    'Nifty 500':'^CRSLDX',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
}

@st.cache_data
def get_stock_info(ticker: str):
    return yf.Ticker(ticker).info

@st.cache_data
def get_stock_data(ticker: str, start, end):
    return yf.download(ticker, start=start, end=end)

@st.cache_data
def get_chart_data(ticker: str):
    return yf.Ticker(ticker).history(period='max')

@st.cache_data
def get_logo(ticker: str):
    return fn.get_logo_from_symbol(ticker)

@st.cache_data
def get_symbol_to_display(stock_list, market):
    return {
        symbol: f"{entry.get('Security', 'Unknown')} ({symbol})"
        for entry in stocks_data_json["market"].get(market, [])
        for symbol in stock_list
        if entry["Symbol"] == symbol
    }


