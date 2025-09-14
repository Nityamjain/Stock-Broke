import streamlit as st
from utils.dataload import market_tickers
import utils.dataload as dl
import utils.functions as fn
import utils.vizuals as vz
import datetime as dt
import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from firebase_admin import credentials, firestore, auth
import firebase_admin
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


def safe_get(info, key, default='N/A'):
    try:
        val = info.get(key, default)
        # Further check for None, empty, or unexpected types
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default

st.set_page_config(page_title='Stock Analysis',
                  page_icon="📊", layout="wide")

st.title("Stock Analysis", anchor="content")

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
if not st.session_state['logged_in']:
    st.switch_page("pages/Login.py")
else:
    with st.sidebar:
        st.header("Navigation")
        st.success(f"Signed in as {st.session_state.usermail}")
        st.text(f'Name: {st.session_state.username}')
        if st.button("Sign Out"):
            signout()

try:
    with st.expander("Select a Stock for Analysis", expanded=True):
        # Replace with your actual data loading logic
        col1, col2 = st.columns([1,1])
        with col1:
            market = st.selectbox("Choose the Market", list(market_tickers))
        with col2:
            # Replace with your actual mapping logic
            stock_mapping = dl.get_display_to_symbol_market(market)
            stock_code = st.selectbox("Choose a stock", list(stock_mapping.keys()))
            selected_symbol = stock_mapping[stock_code]

        col1, col2 = st.columns([1,1])
        today = dt.date.today()
        with col1:
            start_date = st.date_input("Start Date", dt.date(today.year-1, today.month, today.day))
        with col2:
            end_date = st.date_input("End Date", dt.date(today.year, today.month, today.day))
        if end_date < start_date:
            st.error("Enter the Correct Date range")

    ticker = selected_symbol

    # Use cached functions
    info = dl.get_stock_info(ticker)
    data = dl.get_stock_data(ticker, start_date, end_date)
    chart_data = dl.get_chart_data(ticker)

    # Logo and name
    logo_url = fn.get_logo_from_symbol(ticker)  # You can cache this too if needed
    stock_name = safe_get(info, "shortName", stock_code)


    logo_url = fn.get_logo_from_symbol(ticker)
    stock_name = safe_get(info, "shortName", stock_code)

    col1, col2 = st.columns([1,16])
    with col1:
        if logo_url != "":
            st.image(logo_url, width=80)
    with col2:
        st.header(stock_name)

    col1, col2 = st.columns(2)
    with col1:
        industry = safe_get(info, 'industry')
        sector = safe_get(info, 'sector')
        country = safe_get(info, 'country')
        st.markdown(f"**Sector & Industry:** {industry} ({sector})")
        st.markdown(f"**Country:** {country}")
    with col2:
        employees = safe_get(info, 'fullTimeEmployees')
        website = safe_get(info, 'website')
        st.markdown(f"**Full Time Employees:** {employees}")
        # Only display if there's a valid website
        if website != "N/A":
            st.markdown(f"**Website:** [{website}]({website})")
        else:
            st.markdown("**Website:** N/A")

    with st.expander("Company Overview", expanded=True):
        business_summary = safe_get(info, 'longBusinessSummary')
        st.markdown(business_summary)

    # Metrics display
    col1_metrics = {
        'Market Cap': 'marketCap',
        'Beta': 'beta',
        'EPS': 'trailingEps',
        'PE Ratio': 'trailingPE',
        'Devident Yield': 'dividendYield'
    }
    col2_metrics = {
        'Quick Ratio': 'quickRatio',
        'Revenue Per Share': 'revenuePerShare',
        'Profit Margins': 'profitMargins',
        'Debt to Equity': 'debtToEquity',
        'Return on Equity': 'returnOnEquity'
        
    }
    st.subheader("Stats")
    try:
        data = data.xs(ticker, level=1, axis=1)
        data.rename(columns={"Price": "Date"}, inplace=True)
        change_close = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        change_open = data['Open'].iloc[-1] - data['Open'].iloc[-2]
        change_high = data['High'].iloc[-1] - data['High'].iloc[-2]
        change_low = data['Low'].iloc[-1] - data['Low'].iloc[-2]
        if not data.empty and 'Close' in data.columns:
                col1, col2, col3, col4= st.columns(4)
                col1.metric("Close", f"{data['Close'].iloc[-1]:.2f}", f"{change_close:.2f}")
                col2.metric("Open", f"{data['Open'].iloc[-1]:,.2f}",f"{change_open:.2f}")
                col3.metric("High", f"{data['High'].iloc[-1]:,.2f}",f"{change_high:.2f}")
                col4.metric("Low", f"{data['Low'].iloc[-1]:,.2f}",f"{change_low:.2f}")
        st.write(" ")

    except:
        pass            
            
    
    # Robust metric fetching
    col1_data = {label: safe_get(info, key) for label, key in col1_metrics.items()}
    col2_data = {label: safe_get(info, key) for label, key in col2_metrics.items()}
    stats_data = {**col1_data,**col2_data}
    stats_df = pd.DataFrame([start_date])
    df1 = pd.DataFrame(col1_data.values(), index=col1_data.keys(), columns=[''])
    df2 = pd.DataFrame(col2_data.values(), index=col2_data.keys(), columns=[''])
    print(df1,df2)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe((df1))
    with col2:
        st.dataframe(df2)


  
    try:
        
        # Ensure the 'Date' column does not have time
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.date

        if not data.empty and 'Close' in data.columns:
            
        
            col1,col2=st.columns([1,6])
            with col1:
             last_days=st.number_input("**Enter the last number of days to get Histoical data**",10,max_value=365,)
             last_n_days = data.tail(last_days).sort_index(ascending=False).round(3)
             last_n_days.index = pd.to_datetime(last_n_days.index).date
           
            st.write(f'#### Historical Data (Last {last_days} days)')
            st.dataframe(last_n_days    )
        
            
            

            col1,col2,col3 = st.columns([1,1,4])
            with col1:
                   st.download_button(
                    label="Download Historical Dataset",
                    data = last_n_days.to_csv(index=False).encode('utf-8'),
                    file_name=f"{stock_name}_Historical_dataset.csv",
                    mime='text/csv',
                )
            with col2:
                   st.download_button(
                    label="Download Stats Dataset",
                    data = stats_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{stock_name}_Stats_dataset.csv",
                    mime='text/csv',
                )               
        
        else:
            st.write("No historical data available for selected period.")
    except Exception as e:
        st.write("Could not fetch historical data:", str(e))

    
    st.markdown("---")
    st.subheader("**Visuals**")
    st.write("**Choose The Time Range**")

    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12 =st.columns([1,1,1,1.16,1,1,1.16]+ [2]*5) 
    period=''

    with col1:
        if st.button('5D'):
            period ='5d'

    with col2:
        if st.button('1M'):
            period ='1mo'

    with col3:
        if st.button('6M'):
            period ='6mo'

    with col4:
        if st.button('YTD'):
            period ='ytd'

    with col5:
        if st.button('1Y'):
            period ='1y'

    with col6:
        if st.button('5Y'):
            period ='5y'

    with col7:
        if st.button('MAX'):
            period ='max'

    st.write("**Type of Chart Metrics**")
    col1,col2,col3 = st.columns([1,1,4])

    with col1:
        chart_type= st.selectbox('',{'Candle','Line'})

    with col2:
        if chart_type=='Candle':
            Indicators = st.selectbox('',{'RSI','MACD'})
        else:
            Indicators = st.selectbox('',{'RSI','MACD','Moving Average'})




    print( "chart _data:",chart_data.head())
    if period =='':
        
        if chart_type=='Candle' and Indicators == 'RSI':
            st.plotly_chart(vz.candlestick(chart_data,'1y'),use_container_width=True)
            st.plotly_chart(vz.RSI(chart_data,'1y'))
            
        if chart_type=='Candle' and Indicators == 'MACD':
            st.plotly_chart(vz.candlestick(chart_data,'1y'),use_container_width=True)
            st.plotly_chart(vz.MACD(chart_data,'1y')) 

        if chart_type=='Line' and Indicators == 'RSI':
            st.plotly_chart(vz.close_chart(chart_data,'1y'),use_container_width=True)
            st.plotly_chart(vz.RSI(chart_data,'1y'))
            
        if chart_type=='Line' and Indicators == 'MACD':
            st.plotly_chart(vz.close_chart(chart_data,'1y'),use_container_width=True)
            st.plotly_chart(vz.MACD(chart_data,'1y')) 
        
        if chart_type=='Line' and Indicators == 'Moving Average':
            st.plotly_chart(vz.close_chart(chart_data,'1y'),use_container_width=True)
            st.plotly_chart(vz.Moving_average(chart_data,'1y')) 
    else:
        if chart_type == 'Candle' and Indicators =='RSI':
            st.plotly_chart(vz.candlestick(chart_data,period),use_container_width=True)
            st.plotly_chart(vz.RSI(chart_data,period),use_container_width=True)
            
        if chart_type == 'Candle' and Indicators =='MACD':
            st.plotly_chart(vz.candlestick(chart_data,period),use_container_width=True)
            st.plotly_chart(vz.MACD(chart_data,period),use_container_width=True)   
            
        if chart_type == 'Line' and Indicators =='RSI':
            st.plotly_chart(vz.close_chart(chart_data,period),use_container_width=True)
            st.plotly_chart(vz.RSI(chart_data,period),use_container_width=True)
            
        if chart_type == 'Line' and Indicators =='MACD':
            st.plotly_chart(vz.close_chart(chart_data,period),use_container_width=True)
            st.plotly_chart(vz.MACD(chart_data,period),use_container_width=True) 
                
        if chart_type == 'Line' and Indicators =='Moving Average':
            st.plotly_chart(vz.close_chart(chart_data,period),use_container_width=True)
            st.plotly_chart(vz.Moving_average(chart_data,period),use_container_width=True)       


except Exception as e:
    st.write("Check your Internet Connection or data source:", str(e))


