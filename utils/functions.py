import plotly.graph_objects as go
import pandas_ta as pta
import streamlit as st
import yfinance as yf
import requests
import re
import dateutil as du
import datetime as dt
import pandas as pd
from firebase_admin import firestore
# --- CAPM Calculations ---
@st.cache_data
def daily_returns(df):
    df_daily = df.copy()
    for col in df.columns:
        if col != 'Date':
            df_daily[col] = df[col].pct_change() * 100
    return df_daily

@st.cache_data
def normalize_data(df):
    df_norm = df.copy()
    for col in df.columns:
        if col != 'Date':
            df_norm[col] = df[col] / df[col].iloc[0]
    return df_norm

@st.cache_data
def calculate_beta(df, stock,market):
    cov = df[[stock, market]].cov().iloc[0, 1]  # Covariance with market
    var = df[market].var()  # Market variance
    beta = cov / var
    alpha = df[stock].mean() - beta * df[market].mean()
    return beta, alpha


@st.cache_data
def normalize_symbols(symbols) -> list[str]:

    if isinstance(symbols, str):
        return [symbols]
    elif isinstance(symbols, list):
        return [s for s in symbols if isinstance(s, str)]
    else:
        return []



@st.cache_resource
def get_ticker(ticker):
    return yf.Ticker(ticker)


DEFAULT_LOGO = "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"

@st.cache_data
def get_logo_from_symbol(symbol: str) -> str | None:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        website = info.get("website")

        if website:
            domain = (
                website.replace("https://", "")
                .replace("http://", "") 
                .split("/")[0]
                .strip()
            )

            # Securely access your Logo.dev publishable key
            publishable_key = st.secrets["logo_dev"]["publishable_key"]
            logo_url = f"https://img.logo.dev/{domain}?token={publishable_key}"

            response = requests.get(logo_url, timeout=10)
            content_type = response.headers.get("Content-Type", "")
            if response.status_code == 200 and "image" in content_type:
                return logo_url
            print(f"Logo fetch failed with status: {response.status_code}")
    except Exception as e:
        print(f"Logo fetch failed for {symbol}: {e}")

    return DEFAULT_LOGO


@st.cache_data
def filter_data(df,period):
    if period =="1mo":
        date = df.index[-1]+du.relativedelta.relativedelta(months=-1)
    elif period =="5d":
        date = df.index[-1]+du.relativedelta.relativedelta(days=-5)

    elif period =="6mo":
        date = df.index[-1]+du.relativedelta.relativedelta(months=-6)

    elif period =="1y":
        date = df.index[-1]+du.relativedelta.relativedelta(years=-1)
    
    elif period =="5y":
        date = df.index[-1]+du.relativedelta.relativedelta(years=-5)
    elif period =="ytd":
        date = dt.datetime(df.index[-1].year,1,1).strftime('%Y-%m-%d')
    else:
        date= df.index[0]
        
    return df.reset_index()[df.reset_index()['Date']>date]
        
@st.cache_data
def init_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Dark"
    st.sidebar.selectbox(
        "Theme", ["Dark", "Light"],
        index=0 if st.session_state["theme"] == "Dark" else 1,
        key="theme"
    )
    return st.session_state["theme"], 


def render_feedback_form():
    with st.expander("🐛 Found a bug? 💡 Suggest an improvement"):
        with st.form("feedback_form"):
            bug_description = st.text_area("Describe the issue or suggestion", key="bug_description")
            contact_email = st.text_input("Your email (optional)", key="contact_email")
            submitted = st.form_submit_button("Submit")

            if submitted:
                feedback = {
                    "timestamp": dt.datetime.now().isoformat(),
                    "description": bug_description,
                    "email": contact_email
                }
                db.collection("feedbacks").add(feedback)
                st.success("Thanks for your feedback! Logged to database.")

