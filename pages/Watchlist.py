import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from utils.dataload import market_tickers
import utils.dataload as dl

import json 

service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firestore setup
db = firestore.client()
_FIREBASE_READY = True

# Session state initialization
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'usermail' not in st.session_state:
    st.session_state['usermail'] = ''
if 'singedout' not in st.session_state:
    st.session_state['singedout'] = False
if 'singout' not in st.session_state:
    st.session_state['singout'] = False

# Local storage for watchlists
LOCAL_STORE = os.path.join(os.path.dirname(__file__), "..", "data", "watchlists.json")

st.set_page_config(page_title="Watchlist & News", page_icon="⭐", layout="wide")
st.title("Watchlist")

# Authentication functions
def login():
    try:
        user = auth.get_user_by_email(st.session_state.email)
        # Note: Firebase Admin SDK cannot verify passwords directly.
        # For password verification, you would typically use Firebase Authentication client-side SDK.
        # Here, we assume login success if user exists (simplified for Admin SDK).
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.success("Logged in successfully")
    except:
        st.warning("User not found or incorrect credentials. Please sign up or check your email.")

def signup():
    try:
        user = auth.create_user(email=st.session_state.email, password=st.session_state.password, uid=st.session_state.signup_username)
        st.session_state.username = user.uid
        st.session_state.usermail = user.email
        st.session_state.singout = True
        st.session_state.singedout = True
        st.success("Account created and logged in successfully")
    except Exception as e:
        st.error(f"Error creating account: {str(e)}")

def signout():
    st.session_state.singout = False
    st.session_state.singedout = False
    st.session_state.username = ''
    st.session_state.usermail = ''
    st.session_state.pop('watchlist', None)
    st.query_params.clear()
    st.rerun()

# Authentication UI
if not st.session_state['singedout']:
    st.switch_page("pages/Login.py")
else:
    with st.sidebar:
        st.header("Navigation")
        st.success(f"Signed in as {st.session_state.usermail}")
        st.text(f'Name: {st.session_state.username}')
        if st.button("Sign Out"):
            signout()

# Stop if not logged in
if not st.session_state['singedout']:
    st.info("Please log in or sign up to continue.")
    st.stop()

# Watchlist storage functions
def _watchlist_doc_ref(user_id: str):
    return db.collection("watchlists").document(user_id)

def _local_store_read():
    try:
        with open(LOCAL_STORE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _local_store_write(data):
    try:
        os.makedirs(os.path.dirname(LOCAL_STORE), exist_ok=True)
        with open(LOCAL_STORE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_watchlist(user_id: str):
    if _FIREBASE_READY and db is not None:
        try:
            doc = _watchlist_doc_ref(user_id).get()
            if doc.exists:
                return doc.to_dict() or {"symbols": []}
        except Exception:
            pass
    store = _local_store_read()
    return store.get(user_id, {"symbols": []})

def _save_watchlist(user_id: str, data: dict):
    if _FIREBASE_READY and db is not None:
        try:
            _watchlist_doc_ref(user_id).set(data)
            return
        except Exception:
            pass
    store = _local_store_read()
    store[user_id] = data
    _local_store_write(store)

# Fetch market data
def _fetch_metrics(ticker: str):
    t = yf.Ticker(ticker)
    info = t.fast_info or {}
    price = info.get("last_price") or info.get("lastPrice")
    prev_close = info.get("previous_close") or info.get("previousClose")
    market_cap = info.get("market_cap") or info.get("marketCap")
    pe = None
    try:
        pe = t.info.get("trailingPE")
    except Exception:
        pe = None
    change_pct = None
    if price and prev_close and prev_close != 0:
        change_pct = (price - prev_close) / prev_close * 100.0
    return {
        "price": price,
        "change_pct": change_pct,
        "market_cap": market_cap,
        "pe": pe,
    }

def _fetch_news(ticker: str, limit: int = 5):
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        out = []
        for it in items[:limit]:
            out.append({
                "title": it.get("title"),
                "publisher": it.get("publisher"),
                "link": it.get("link"),
                "providerPublishTime": it.get("providerPublishTime"),
                "summary": it.get("summary") or "",
            })
        return out
    except Exception:
        return []

# Plotting functions
def _plot_price_chart(ticker: str):
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=180)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=220)
        return fig
    except Exception:
        return None

# Load watchlist
user_id = st.session_state.username
watch = st.session_state.get("watchlist") or _load_watchlist(user_id)
if "symbols" not in watch:
    watch["symbols"] = []
if "notes" not in watch:
    watch["notes"] = {}
st.session_state["watchlist"] = watch

# Watchlist UI
with st.expander("Select a Stock for Watchlist", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        market = st.selectbox("Choose the Market", list(market_tickers))
    with col2:
        stock_mapping = dl.get_display_to_symbol_market(market)
        stock_code = st.selectbox("Choose a stock", list(stock_mapping.keys()))
        selected_symbol = stock_mapping[stock_code]
    with col3:
        if st.button("Add Selected"):
            if selected_symbol not in watch["symbols"]:
                watch["symbols"].append(selected_symbol)
                _save_watchlist(user_id, watch)
                st.success(f"Added {selected_symbol} to watchlist")
                st.rerun()
            else:
                st.info(f"{selected_symbol} is already in your watchlist")

# Search/Add by ticker
st.subheader("Or Search by Ticker/Name")
col_sa, col_sb = st.columns([3, 1])
with col_sa:
    query = st.text_input("Ticker or name", placeholder="e.g., AAPL or Apple")
with col_sb:
    add_clicked = st.button("Add")

def _resolve_symbol(q: str):
    if not q:
        return None
    q = q.strip().upper()
    try:
        t = yf.Ticker(q)
        info = t.fast_info
        if info is not None and (info.get("last_price") or info.get("lastPrice")) is not None:
            return q
    except Exception:
        pass
    try:
        srch = yf.search(q)
        if srch and isinstance(srch, list):
            for r in srch:
                sym = r.get("symbol")
                if sym:
                    return sym
    except Exception:
        pass
    return None

if add_clicked and query:
    sym = _resolve_symbol(query)
    if sym:
        if sym not in watch["symbols"]:
            watch["symbols"].append(sym)
            _save_watchlist(user_id, watch)
            st.success(f"Added {sym} to watchlist")
            st.rerun()
        else:
            st.info(f"{sym} is already in your watchlist")
    else:
        st.warning("Could not find that symbol")

st.markdown("---")

# Watchlist controls
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    sort_by = st.selectbox("Sort by", ["Symbol", "Change %", "Market Cap"], index=1)
with c2:
    filter_text = st.text_input("Filter", placeholder="type to filter symbols")
with c3:
    show_news = st.checkbox("Show News", value=True)

symbols = list(watch.get("symbols", []))
if filter_text:
    ft = filter_text.strip().lower()
    symbols = [s for s in symbols if ft in s.lower()]

rows = []
metrics_map = {}
for s in symbols:
    m = _fetch_metrics(s)
    metrics_map[s] = m
    rows.append([
        s,
        m.get("price"),
        m.get("change_pct"),
        m.get("market_cap"),
        m.get("pe"),
    ])

df_tbl = pd.DataFrame(rows, columns=["Symbol", "Price", "Change %", "Market Cap", "PE"])
if sort_by == "Symbol":
    df_tbl = df_tbl.sort_values("Symbol")
elif sort_by == "Change %":
    df_tbl = df_tbl.sort_values("Change %", ascending=False, na_position='last')
elif sort_by == "Market Cap":
    df_tbl = df_tbl.sort_values("Market Cap", ascending=False, na_position='last')

st.subheader("Watchlist")

# Symbol to market mapping
_symbol_to_market_cache = {}
def _resolve_market_for_symbol(sym: str):
    if sym in _symbol_to_market_cache:
        return _symbol_to_market_cache[sym]
    try:
        for mk in list(market_tickers):
            try:
                mapping = dl.get_display_to_symbol_market(mk)
                for disp, ss in mapping.items():
                    if ss == sym:
                        _symbol_to_market_cache[sym] = mk
                        return mk
            except Exception:
                continue
    except Exception:
        pass
    _symbol_to_market_cache[sym] = "Unknown"
    return "Unknown"

# Group symbols by market
market_to_symbols = {}
for s in symbols:
    mk = _resolve_market_for_symbol(s)
    market_to_symbols.setdefault(mk, []).append(s)

combined_rows = []
for mk, syms in market_to_symbols.items():
    if not syms or mk == "Unknown":
        continue
    rows_m = []
    for s in syms:
        m = metrics_map.get(s, {})
        logo_url = None
        try:
            logo_url = dl.get_logo(s)
        except Exception:
            logo_url = None
        row = {
            "Logo": logo_url,
            "Symbol": s,
            "Price": m.get("price"),
            "Change %": m.get("change_pct"),
            "Market Cap": m.get("market_cap"),
            "PE": m.get("pe"),
        }
        rows_m.append(row)
        combined_rows.append({**row, "Market": mk})

    if rows_m:
        st.markdown(f"#### {mk}")
        df_m = pd.DataFrame(rows_m)
        st.dataframe(
            df_m,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Logo": st.column_config.ImageColumn(label="", width="small"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                "Market Cap": st.column_config.NumberColumn(),
                "PE": st.column_config.NumberColumn(format="%.2f"),
            },
        )

# Show unknown symbols
unknown_syms = market_to_symbols.get("Unknown", [])
if unknown_syms:
    rows_u = []
    for s in unknown_syms:
        m = metrics_map.get(s, {})
        logo_url = None
        try:
            logo_url = dl.get_logo(s)
        except Exception:
            logo_url = None
        rows_u.append({
            "Logo": logo_url,
            "Symbol": s,
            "Price": m.get("price"),
            "Change %": m.get("change_pct"),
            "Market Cap": m.get("market_cap"),
            "PE": m.get("pe"),
        })
        combined_rows.append({
            "Logo": logo_url,
            "Symbol": s,
            "Price": m.get("price"),
            "Change %": m.get("change_pct"),
            "Market Cap": m.get("market_cap"),
            "PE": m.get("pe"),
            "Market": "Unknown",
        })
    if rows_u:
        st.markdown("#### Other")
        st.dataframe(pd.DataFrame(rows_u), use_container_width=True, hide_index=True)

# Download watchlist
if combined_rows:
    df_all = pd.DataFrame(combined_rows)
    csv_df = df_all.drop(columns=["Logo"], errors="ignore")
    st.download_button(
        label="Download Watchlist (CSV)",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="watchlist.csv",
        mime="text/csv",
    )

# News section (if enabled)
if show_news and symbols:
    for s in symbols:
        news = _fetch_news(s)
        if news:
            st.subheader(f"News for {s}")
            for item in news:
                st.markdown(f"**{item['title']}** ({item['publisher']})")
                st.markdown(item['summary'])
                st.markdown(f"[Read more]({item['link']})")

                st.markdown("---")

