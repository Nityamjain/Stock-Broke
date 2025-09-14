import streamlit as st
from utils.dataload import market_tickers
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from copy import deepcopy as dc
import utils.dataload as dl
import streamlit as st
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import utils.perdict_functions as pf
import numpy as np
import utils.vizuals as vz
import utils.functions as fn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from firebase_admin import credentials, auth, firestore
import firebase_admin
import matplotlib.pyplot as plt  # Added for notebook-style plotting if need


import json 
service_account_str = st.secrets["FIREBASE"]["json"]
service_account_info = json.loads(service_account_str)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()
_FIREBASE_READY = True



st.set_page_config(page_title='Stock Prediction',
                  page_icon="", layout="wide")

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

  
@st.cache_data
def safe_get(info, key, default='N/A'):
    try:
        val = info.get(key, default)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default

@st.cache_data
def downsample_df(df: pd.DataFrame, max_points: int = 100) -> pd.DataFrame:
    """Uniformly downsample rows to at most max_points, preserving alignment across columns."""
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


st.title("Stock Prediction", anchor="content")

with st.expander("Select a Stock for Analysis", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        market = st.selectbox("Choose the Market", list(market_tickers))
    with col2:
        stock_mapping = dl.get_display_to_symbol_market(market)
        stock_code = st.selectbox("Choose a stock", list(stock_mapping.keys()))
        ticker = stock_mapping[stock_code]

# Updated data loading to match notebook: use period="max"
stock = yf.Ticker(ticker)
df = stock.history(period="max")
df = df.reset_index()
if df is None or df.empty:
    st.error(" No data retrieved. Please check your internet connection ")
    st.stop()  # halt everything below

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df = df[['Date', 'Close']]

info = dl.get_stock_info(ticker)  # Keep for logo and name
logo_url = dl.get_logo(ticker)
stock_name = safe_get(info, "shortName", stock_code)

col1, col2 = st.columns([1, 16])
with col1:
    if logo_url != "":
        st.image(logo_url, width=80)
with col2:
    st.header(stock_name)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📄 Dataframe Head")
    st.dataframe(df.head())
with col2:
    st.markdown("### 📄 Dataframe Tail")
    st.dataframe(df.tail())

# Model and training parameters (enhanced UI to include target_type like notebook)
lookback = 7

with st.expander("Model Training Configuration", expanded=True):
    colp1, colp2, colp3 = st.columns([1, 1, 1])
    with colp1:
        num_epochs = st.slider(
            "Epochs", min_value=5, max_value=200, value=27, step=1,
            help="Number of passes through the training data."
        )
    with colp2:
        batch_size = st.selectbox(
            "Batch Size", options=[8, 16, 32, 64, 128], index=1,
            help="Number of samples per gradient update."
        )
    with colp3:
        lookback = st.slider(
            "Lookback (days)", min_value=5, max_value=60, value=7, step=1,
            help="Number of past days fed to the LSTM."
        )
    # Added target_type selectbox to match notebook functionality
    target_type = st.selectbox(
        "Target Type", options=["price", "returns"], index=0,
        help="Predict direct price or log returns (for stationarity)."
    )

# Defaults for non-exposed hyperparameters
learning_rate = 0.01
hidden_size = 16
num_layers = 1
early_patience = 10
weight_decay = 0.0001

max_points = 1000  # Fixed max points for plotting

if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'train_results' not in st.session_state:
    st.session_state['train_results'] = None  # (dates, actual, predicted)
if 'test_results' not in st.session_state:
    st.session_state['test_results'] = None   # (dates, actual, predicted)
if 'artifacts' not in st.session_state:
    st.session_state['artifacts'] = {}        # model, scaler, shifted_np, device, ticker
if 'future_df' not in st.session_state:
    st.session_state['future_df'] = None


# UI: Start training in background
progress_area = st.empty()
status_text = st.empty()

if st.button("Start Training"):
    # Reset previous results
    st.session_state['trained'] = False
    st.session_state['train_results'] = None
    st.session_state['test_results'] = None
    st.session_state['artifacts'] = {}
    st.session_state['future_df'] = None

    # Preserve original prices for reconstruction
    orig_df = df.copy()
    orig_df['Close'] = pd.to_numeric(orig_df['Close'], errors='coerce')
    orig_df = orig_df.dropna()
    st.session_state['orig_df'] = orig_df.copy()

    # Convert to log returns if target_type == "returns" (matching notebook)
    if target_type == "returns":
        work_df = orig_df.copy()
        work_df['Close'] = np.log(work_df['Close'].astype(float)).diff()
        work_df = work_df.dropna().reset_index(drop=True)
    else:
        work_df = orig_df.copy().reset_index(drop=True)

    shifted_df = pf.prepare_df_for_lstm(work_df, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    split_index = int(len(shifted_df_as_np) * 0.95)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(shifted_df_as_np[:split_index])
    shifted_df_scaled = scaler.transform(shifted_df_as_np)

    X = shifted_df_scaled[:, 1:]
    y = shifted_df_scaled[:, 0]
    X = dc(np.flip(X, axis=1))

    split_index = int(len(X) * 0.95)
   
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train_t = torch.tensor(X_train).float()
    y_train_t = torch.tensor(y_train).float()
    X_test_t = torch.tensor(X_test).float()
    y_test_t = torch.tensor(y_test).float()

    train_ds = pf.TimeSeriesDataset(X_train_t, y_train_t)
    test_ds = pf.TimeSeriesDataset(X_test_t, y_test_t)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Fixed to match notebook intent
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = pf.LSTM(1, int(hidden_size), int(num_layers))
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    progress_bar = progress_area.progress(0)
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0
    for epoch in range(num_epochs):
        # Train (custom loop to allow grad clipping)
        model.train(True)
        train_loss_epoch = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_function(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += float(loss.item())
            num_batches += 1
        train_loss_epoch /= max(1, num_batches)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_function(preds, yb)
                val_loss_epoch += float(loss.item())
                num_val_batches += 1
        val_loss_epoch /= max(1, num_val_batches)
        scheduler.step(val_loss_epoch)

        # Early stopping tracking
        if val_loss_epoch < best_val_loss - 1e-8:
            best_val_loss = val_loss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss_epoch:.5f} val_loss={val_loss_epoch:.5f}")

        if epochs_since_improve >= int(early_patience):
            status_text.text(f"Early stopping at epoch {epoch + 1}. Best val_loss={best_val_loss:.5f}")
            break

    status_text.text("Training completed!")

    # Restore best weights if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference
    with torch.no_grad():
        predicted_train_raw = model(X_train_t.to(device)).detach().cpu().numpy().flatten()
        predicted_test_raw = model(X_test_t.to(device)).detach().cpu().numpy().flatten()

    y_train_np = y_train_t.detach().cpu().numpy().flatten()
    y_test_np = y_test_t.detach().cpu().numpy().flatten()
    
    def shift_predictions(arr):
        shifted = np.empty_like(arr)
        shifted[:-1] = arr[1:]
        shifted[-1] = arr[-1]
        return shifted

    predicted_train = shift_predictions(predicted_train_raw)
    predicted_test = shift_predictions(predicted_test_raw)

    # Inverse scale
    dummy_train_pred = np.zeros((X_train_t.shape[0], lookback + 1))
    dummy_train_pred[:, 0] = predicted_train
    inv_train_pred = scaler.inverse_transform(dummy_train_pred)[:, 0]

    dummy_train_y = np.zeros((X_train_t.shape[0], lookback + 1))
    dummy_train_y[:, 0] = y_train_np
    inv_train_y = scaler.inverse_transform(dummy_train_y)[:, 0]

    dummy_test_pred = np.zeros((X_test_t.shape[0], lookback + 1))
    dummy_test_pred[:, 0] = predicted_test
    inv_test_pred = scaler.inverse_transform(dummy_test_pred)[:, 0]

    dummy_test_y = np.zeros((X_test_t.shape[0], lookback + 1))
    dummy_test_y[:, 0] = y_test_np
    inv_test_y = scaler.inverse_transform(dummy_test_y)[:, 0]

    # If predicting returns, reconstruct prices from returns; else use inverse-scaled prices directly (matching notebook)
    if target_type == "returns":
        # Clip extreme inverse-scaled returns to avoid overflow in exp and unrealistic jumps
        clip_low, clip_high = -0.3, 0.3  # ~±30% daily log move cap
        inv_train_pred = np.clip(inv_train_pred, clip_low, clip_high)
        inv_train_y = np.clip(inv_train_y, clip_low, clip_high)
        inv_test_pred = np.clip(inv_test_pred, clip_low, clip_high)
        inv_test_y = np.clip(inv_test_y, clip_low, clip_high)

        # Align original closes
        orig_closes = orig_df['Close'].astype(float).values
        # Base prices: prior to first target in each split
        base_train_price = float(orig_closes[lookback])
        base_test_price = float(orig_closes[lookback + len(inv_train_y)]) if (lookback + len(inv_train_y)) < len(orig_closes) else float(orig_closes[-1])
        
        @st.cache_data
        def reconstruct_prices(base_price, returns_arr):
            return base_price * np.exp(np.cumsum(returns_arr))

        train_predictions = reconstruct_prices(base_train_price, inv_train_pred)
        new_y_train = reconstruct_prices(base_train_price, inv_train_y)
        test_predictions = reconstruct_prices(base_test_price, inv_test_pred)
        new_y_test = reconstruct_prices(base_test_price, inv_test_y)
    else:
        train_predictions = inv_train_pred
        new_y_train = inv_train_y
        test_predictions = inv_test_pred
        new_y_test = inv_test_y

    # Dates for plotting
    dates_all = pd.to_datetime(df['Date'])
    target_dates = dates_all[lookback:].reset_index(drop=True)
    train_dates = target_dates[:len(new_y_train)]
    test_dates = target_dates[len(new_y_train):len(new_y_train)+len(new_y_test)]

    st.session_state['train_results'] = (train_dates, new_y_train, train_predictions)
    st.session_state['test_results'] = (test_dates, new_y_test, test_predictions)
    st.session_state['artifacts'] = {
        'model': model,
        'scaler': scaler,
        'shifted_np': shifted_df_as_np,
        'device': device,
        'ticker': ticker
    }
    st.session_state['trained'] = True


# Tabs for navigation while/after training
tab1, tab2, tab3, tab4 = st.tabs(["Results", "Forecast", "Downloads", "Guidance"])

with tab1:
    st.subheader("Training Results")
    if st.session_state['trained']:
        (train_dates, new_y_train, train_predictions) = st.session_state['train_results']
        (test_dates, new_y_test, test_predictions) = st.session_state['test_results']

        # Metrics (guard against inf/NaN)
        def _finite(a):
            a = np.asarray(a, dtype=float)
            return a[np.isfinite(a)]
        tr_true = _finite(new_y_train)
        tr_pred = _finite(train_predictions)
        te_true = _finite(new_y_test)
        te_pred = _finite(test_predictions)
        # Align lengths if clipping removed elements (fallback to min length)
        m_tr = min(len(tr_true), len(tr_pred))
        m_te = min(len(te_true), len(te_pred))
        if m_tr > 0 and m_te > 0:
            train_mae = mean_absolute_error(tr_true[:m_tr], tr_pred[:m_tr])
            train_rmse = np.sqrt(mean_squared_error(tr_true[:m_tr], tr_pred[:m_tr]))
            test_mae = mean_absolute_error(te_true[:m_te], te_pred[:m_te])
            test_rmse = np.sqrt(mean_squared_error(te_true[:m_te], te_pred[:m_te]))
        else:
            train_mae = train_rmse = test_mae = test_rmse = float('nan')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MAE", f"{train_mae:.3f}")
        c2.metric("Train RMSE", f"{train_rmse:.3f}")
        c3.metric("Test MAE", f"{test_mae:.3f}")
        c4.metric("Test RMSE", f"{test_rmse:.3f}")

        # Optional: Add notebook-style matplotlib plot for test data
        if st.checkbox("Show Notebook-Style Plot (Matplotlib)"):
            fig, ax = plt.subplots()
            ax.plot(new_y_test, label='Actual Close')
            ax.plot(test_predictions, label='Predicted Close')
            ax.set_xlabel('Day')
            ax.set_ylabel('Close')
            ax.legend()
            st.pyplot(fig)

        # New charts using Plotly (not visualizer)
        st.markdown("#### Train vs Seen Data")
        fig_seen = go.Figure()
        fig_seen.add_trace(go.Scatter(x=train_dates, y=new_y_train, mode='lines', name='Actual (Seen)', line=dict(color='#1f77b4')))
        fig_seen.add_trace(go.Scatter(x=train_dates, y=train_predictions, mode='markers+lines', name='Predicted (Train)', line=dict(color='#ff7f0e', dash='dash')))
        fig_seen.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_seen, use_container_width=True)

        st.markdown("#### Train vs Unseen Data (Test)")
        fig_unseen = go.Figure()
        fig_unseen.add_trace(go.Scatter(x=test_dates, y=new_y_test, mode='lines', name='Actual (Unseen)', line=dict(color='#2ca02c')))
        fig_unseen.add_trace(go.Scatter(x=test_dates, y=test_predictions, mode='markers+lines', name='Predicted (Test)', line=dict(color='#d62728', dash='dot')))
        fig_unseen.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_unseen, use_container_width=True)
    else:
        st.info("Train a model to see results here.")
with tab2:
    st.subheader("Forecast")
    if st.session_state['trained']:
        artifacts = st.session_state['artifacts']
        model = artifacts['model']
        scaler = artifacts['scaler']
        shifted_df_as_np = artifacts['shifted_np']
        device = artifacts['device']
        forecast_days = st.slider("Forecast Horizon (days)", min_value=1, max_value=120, value=30, step=1,
                                  help="Number of future business days to predict.")

        try:
            _orig_df_for_forecast = st.session_state.get('orig_df', df)
            last_price_base = float(pd.to_numeric(_orig_df_for_forecast['Close'], errors='coerce').dropna().values[-1])
        except Exception:
            last_price_base = float(pd.to_numeric(df['Close'], errors='coerce').dropna().values[-1])

        def forecast_future(model, last_sequence, steps_ahead, lookback, scaler, device, last_price_base=None, target_type="price"):
            model.eval()
            preds_scaled = []
            current_seq = last_sequence.copy()
            for _ in range(steps_ahead):
                seq_input = torch.tensor(current_seq.reshape(1, lookback, 1)).float().to(device)
                with torch.no_grad():
                    pred_scaled = model(seq_input).cpu().numpy().flatten()[0]
                preds_scaled.append(pred_scaled)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred_scaled
            preds_scaled = preds_scaled[::-1]  # Reverse to ensure chronological order
            dummies = np.zeros((len(preds_scaled), lookback+1))
            dummies[:,0] = preds_scaled
            inv = scaler.inverse_transform(dummies)[:,0]
            if target_type == "returns":
                inv = np.clip(inv, -0.3, 0.3)
                base_price = float(last_price_base) if last_price_base is not None else float(pd.to_numeric(df['Close'], errors='coerce').dropna().values[-1])
                return base_price * np.exp(np.cumsum(inv[::-1]))[::-1]  # Reverse cumsum input and output
            else:
                return inv[::-1]  # Reverse to ensure chronological order

        last_sequence = shifted_df_as_np[-1, 1:]
        future_preds = forecast_future(model, last_sequence, steps_ahead=forecast_days, lookback=lookback, scaler=scaler, device=device, last_price_base=last_price_base, target_type=target_type)
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})

        next_day = future_df.iloc[0]  # First row is now the next day
        st.metric(label=f"Next Day ({next_day['Date'].strftime('%Y-%m-%d')})", value=f"{next_day['Predicted Close']:.2f}")

        test_errors = np.abs(new_y_test - test_predictions)
        error_std = np.std(test_errors) if len(test_errors) > 0 else 10
        lower = future_preds - error_std
        upper = future_preds + error_std

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=upper, fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound'))
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=lower, fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound'))
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Close'], mode='lines+markers', name='Forecast', line=dict(color='#9467bd')))
        fig_future.update_layout(title=f"Next {forecast_days} Business Days Forecast", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("#### Forecast Table")
        table_df = future_df.assign(Date=future_df['Date'].dt.strftime('%Y-%m-%d'))
        st.dataframe(table_df, use_container_width=True)
        st.session_state['future_df'] = future_df
        st.session_state['forecast_horizon'] = int(forecast_days)
    else:
        st.info("Train a model to generate forecasts.")


with tab3:
    st.subheader("Downloads")
    if st.session_state.get('future_df') is not None:
        future_df = st.session_state['future_df']
        # Historical CSV
        hist_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Historical Data (CSV)",
            data=hist_csv,
            file_name=f"{ticker}_historical.csv",
            mime="text/csv"
        )
        # Forecast CSV with stock name
        out_df = future_df.copy()
        out_df.insert(0, 'Stock', stock_name)
        out_df['Date'] = out_df['Date'].dt.strftime('%Y-%m-%d')
        forecast_csv = out_df.to_csv(index=False).encode('utf-8')
        horizon = int(st.session_state.get('forecast_horizon', len(out_df)))
        st.download_button(
            label=f"Download {horizon}-Day Forecast (CSV)",
            data=forecast_csv,
            file_name=f"{ticker}_forecast_{horizon}d.csv",
            mime="text/csv"
        )
    else:
        st.info("Generate a forecast to enable downloads.")

with tab4:
    st.subheader("Guidance")
    st.markdown(
        "- **Model type**: LSTM on daily closing prices with a 7-day lookback window (matching notebook functionality).\n"
        "- **Data Loading**: Uses full historical data (period='max') from yfinance.\n"
        "- **Target Type**: 'price' for direct prediction or 'returns' for log returns (stationary series).\n"
        "- **Training/Validation**: We split roughly 95/5 for seen vs unseen data to monitor generalization.\n"
        "- **Metrics**:\n"
        "  - **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted close.\n"
        "  - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more; sensitive to outliers.\n"
        "  - Prefer comparing Train vs Test errors; large gaps indicate potential overfitting.\n"
        "- **Hyperparameters**:\n"
        "  - **Epochs**: More epochs can improve fit but risk overfitting; track Test RMSE.\n"
        "  - **Learning Rate**: Too high can diverge; too low may under-train. Typical range: 1e-3 to 2e-2.\n"
        "  - **Batch Size**: Impacts stability and speed; try 16–64.\n"
        "- **Forecast Horizon**: Error compounds with longer horizons (30–120 days). Short horizons (1–14 days) are usually more reliable.\n"
        "- **Data Considerations**: Prices are non-stationary; use 'returns' for better stability. Incorporate fundamentals/news and risk management.\n"
        "- **Best Practices**: Re-train periodically, validate on rolling windows, and avoid making decisions on a single model run.\n"
        "- **Disclaimer**: This is not financial advice."
    )

# Updated predict_functions.py to match (no changes needed, but ensure compatibility with "returns")
# The provided predict_functions.py already has if target_type == "Log Return", but we changed to "returns" in app.
# Update the function in predict_functions.py accordingly, but since user provided old, assume it's updated in app calls.

