# Add this to the top of Home.py, Login.py, and all files in pages/
import streamlit as st
from utils.dataload import market_tickers,region_market_stocks
import utils.dataload as dl
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import utils.functions as fn
import utils.vizuals as vz
import time
import firebase_admin
from firebase_admin import credentials, auth, firestore
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



# Streamlit page setup
st.set_page_config(page_title='CAPM Analysis',
                  page_icon="💰", layout="wide")


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
            
st.title("Capital Asset Pricing Model", anchor="content")



# --- UI ---
col1, col2 = st.columns([1, 1])
with col1:
    region = st.selectbox("🌍 Choose Region", list(region_market_stocks.keys()))
with col2:
    market = st.selectbox("📈 Choose Market Index", list(region_market_stocks[region].keys()))
    default_stocks = region_market_stocks[region][market]

col1, col2 = st.columns([2, 1])
with col1:
    display_to_symbol = dl.get_display_to_symbol(region, market)
    stock_display_list = list(display_to_symbol.keys())
    default_display = stock_display_list[:2]

    selected_display_names = st.multiselect(
        "📊 Choose up to 4 stocks",
        options=stock_display_list,
        default=default_display,
        max_selections=4
    )

    stock_list = [display_to_symbol[name] for name in selected_display_names]


with col2:
    year = st.number_input('📅 Number of Years', min_value=1, max_value=10, value=5)



# Date Range
end = dt.date.today()
start = dt.date(end.year - year, end.month, end.day)

# Cache data downloads

@st.cache_data
def download_data(ticker, start, end, retries=3):
    for _ in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                return data
            time.sleep(2)
        except Exception as e:
            st.warning(f"⚠ Failed to download {ticker}: {str(e)}")
            time.sleep(2)
    return pd.DataFrame()

# Download Benchmark Data
benchmark_ticker = market_tickers[market]

benchmark_df = download_data(benchmark_ticker, start, end)
if benchmark_df.empty:
    st.error(f"⚠ No benchmark data found for {market}")
    st.stop()

# Reset index and flatten columns
benchmark_df = benchmark_df.reset_index()  # Convert index to columns
if isinstance(benchmark_df.columns, pd.MultiIndex):
    benchmark_df.columns = benchmark_df.columns.get_level_values(0)  # Flatten MultiIndex to first level
elif isinstance(benchmark_df.columns, pd.Index) and benchmark_df.columns.nlevels > 1:
    benchmark_df.columns = benchmark_df.columns.get_level_values(0)
# Select and rename columns
if 'Date' in benchmark_df.columns and 'Close' in benchmark_df.columns:
    benchmark_df = benchmark_df[['Date', 'Close']]
    benchmark_df.rename(columns={'Close': market}, inplace=True)
else:
    st.error(f"⚠ Expected columns 'Date' and 'Close' not found in benchmark data for {market}")
    st.stop()
benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])

# Download Stock Data
stocks_df = pd.DataFrame()
dates = None
for stock in stock_list:
    data = download_data(stock, start, end)
    if not data.empty:
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex
        if 'Date' not in data.columns:
            st.error(f" No 'Date' column in data for {stock}")
            st.stop()
        data['Date'] = pd.to_datetime(data['Date'])
        if dates is None:
            dates = data['Date']
            stocks_df['Date'] = dates
        stocks_df[stock] = data.set_index('Date')['Close'].reindex(dates).values
    else:
        st.warning(f" No data found for {stock}")

if stocks_df.empty or 'Date' not in stocks_df.columns:
    st.error(" No valid stock data available. Please check your selections or internet connection.")
    st.stop()

# Merge on Date
try:
    merged_df = pd.merge(stocks_df, benchmark_df, on='Date', how='inner')
    if merged_df.empty:
        st.error(" No overlapping dates between stocks and benchmark data.")
        st.stop()
except Exception as e:
    st.error(f" Merge failed: {str(e)}")
    st.stop()

merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.strftime('%Y-%m-%d')
# Build symbol → "Security (Symbol)" mapping
symbol_to_display = dl.get_symbol_to_display(stock_list,market)
        
# Rename stock columns to "Security (Symbol)"
merged_df.rename(columns={
    col: symbol_to_display.get(col, col)
    for col in merged_df.columns
    if col not in ['Date', market]
}, inplace=True)

# Display stock logos
cols = st.columns(len(stock_list))
for i, symbol in enumerate(stock_list):
    with cols[i]:
        logo_url = dl.get_logo(symbol)
        print(f'symbol:{symbol}')
        if logo_url:
            st.image(logo_url, width=80)
        else:
            st.caption(f" {symbol}")
                
# Display DataFrames
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📄 Dataframe Head")
    st.dataframe(merged_df.head())
with col2:
    st.markdown("### 📄 Dataframe Tail")
    st.dataframe(merged_df.tail())

merge_csv = merged_df.to_csv(index=False)




# Plotting
st.markdown("---")
st.markdown(" ### **Price of Stocks**")
st.plotly_chart(vz.interactive_plot(merged_df), use_container_width=True)
st.markdown(" ")
st.markdown("### **Normalized Price of Stocks**")
st.plotly_chart(vz.interactive_plot(fn.normalize_data(merged_df)), use_container_width=True)
st.markdown("---")
# Calculate daily returns
stocks_daily_return = fn.daily_returns(merged_df)

# Calculate beta and alpha
beta = {}
alpha = {}
for i in stocks_daily_return.columns:
    if i != 'Date' and i != market:
        b, a = fn.calculate_beta(stocks_daily_return, i,market)
        beta[i] = b
        alpha[i] = a

# Create beta DataFrame
beta_df = pd.DataFrame({
    'Stock': [symbol_to_display.get(s, s) for s in beta.keys()],
    'Beta': [round(i, 2) for i in beta.values()]
})


# Display beta
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('### Calculated Beta Value')
    st.dataframe((beta_df))
    
# Calculate expected returns using CAPM
rf = 0  # Risk-free rate
rm = stocks_daily_return[market].mean() * 252  # Annualized market return

return_df = pd.DataFrame(columns=['Stock', 'Expected Return'])
for stock, beta_val in beta.items():
    expected_return = rf + beta_val * (rm - rf)
    display_name = symbol_to_display.get(stock, stock)
    return_df.loc[len(return_df)] = [display_name, round(expected_return, 2)]

# Display expected returns
with col2:
    st.markdown('### Calculated Return using CAPM')
    st.dataframe(return_df)
    




col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

# Download button for the merged dataset
with col1:
    st.download_button(
        label="Download Dataset",
        data=merged_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{stock_list[0] if len(stock_list) > 0 else 'selected_stocks'}_dataset.csv",
        mime='text/csv',
    )

# Download button for the betas dataset
with col2:
    st.download_button(
        label="Download Betas",
        data=beta_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{stock_list[0] if len(stock_list) > 0 else 'selected_stocks'}_betas.csv",
        mime='text/csv',
    )

# Download button for the returns dataset
with col3:
    st.download_button(
        label="Download Returns",
        data=return_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{stock_list[0] if len(stock_list) > 0 else 'selected_stocks'}_returns.csv",
        mime='text/csv',
    )




st.markdown("---")
st.markdown("### CAPM Basics and Returns")
st.markdown(
    """
    - **What is CAPM?**: The Capital Asset Pricing Model links a security's expected return to its systematic risk (beta) relative to a market benchmark.
    - **CAPM Formula**: `E[R_i] = R_f + β_i (E[R_m] - R_f)`
      - **E[R_i]**: Expected return of asset i
      - **R_f**: Risk‑free rate (e.g., short-term Treasury)
      - **E[R_m]**: Expected return of the market (benchmark index)
      - **β_i**: Sensitivity of the stock’s returns to market returns
    - **Interpreting β (Beta)**:
      - **β = 1.0**: Moves roughly with the market
      - **β > 1.0**: More volatile than the market (higher risk/return)
      - **β < 1.0**: Less volatile than the market (lower risk/return)
    - **Returns**:
      - Daily returns are computed from percentage changes in closing prices.
      - Beta is estimated by regressing stock returns on market returns over the selected period.
      - Annualized market return ≈ average daily return × 252 trading days.
    - **Notes**:
      - CAPM is a simplification; actual returns may be impacted by additional factors (size, value, momentum, liquidity) and idiosyncratic risks.
      - Ensure overlapping dates and a suitable benchmark when interpreting β and expected returns.
    """
)

