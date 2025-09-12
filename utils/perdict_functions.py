from copy import deepcopy as dc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

def safe_get(info, key, default='N/A'):
    """Safely get a value from a dictionary, returning a default if it's missing, None, or empty."""
    try:
        val = info.get(key, default)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default

def prepare_df_for_lstm(df, n_step):
    """
    Prepares a DataFrame for LSTM by creating n_step lagged columns of the 'Close' price.
    """
    df = dc(df)
    df.set_index('Date', inplace=True)
    
    # Create lagged columns correctly
    for i in range(1, n_step + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
        
    df.dropna(inplace=True)
    
    return df

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LSTM(nn.Module):
    """LSTM model for time series prediction."""
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, device, train_loader, test_loader, loss_function, optimizer, scheduler, num_epochs, early_patience, status_text, progress_bar):
    """
    Main training loop with validation, early stopping, and Streamlit progress updates.
    """
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train(True)
        train_loss_epoch = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_function(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= max(1, len(train_loader))

        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_function(preds, yb)
                val_loss_epoch += loss.item()
        val_loss_epoch /= max(1, len(test_loader))
        scheduler.step(val_loss_epoch)

        # Early stopping and best model tracking
        if val_loss_epoch < best_val_loss - 1e-8:
            best_val_loss = val_loss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # Update Streamlit UI
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss_epoch:.5f} val_loss={val_loss_epoch:.5f}")

        if epochs_since_improve >= early_patience:
            status_text.text(f"Early stopping at epoch {epoch + 1}. Best val_loss={best_val_loss:.5f}")
            break
            
    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def shift_predictions(arr):
    """Shifts predictions by one step to align with actual values for evaluation."""
    shifted = np.empty_like(arr)
    shifted[:-1] = arr[1:]
    shifted[-1] = arr[-1]  # Keep last prediction for continuity
    return shifted

def _finite(a):
    """Filters an array to return only finite values."""
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]

@st.cache_data
def reconstruct_prices(base_price, returns_arr):
    """Reconstructs absolute prices from a base price and an array of log returns."""
    return base_price * np.exp(np.cumsum(returns_arr))

def forecast_future(_model, last_sequence, steps_ahead, lookback, _scaler, device, target_type, last_price_base):
    """
    Generates future predictions by iteratively feeding predictions back into the model.
    """
    _model.eval()
    preds_scaled = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps_ahead):
        seq_input = torch.tensor(current_seq.reshape(1, lookback, 1)).float().to(device)
        with torch.no_grad():
            pred_scaled = _model(seq_input).cpu().numpy().flatten()[0]
        preds_scaled.append(pred_scaled)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred_scaled
        
    dummies = np.zeros((len(preds_scaled), lookback + 1))
    dummies[:, 0] = preds_scaled
    inv = _scaler.inverse_transform(dummies)[:, 0]
    
    if target_type == "Log Return":
        inv = np.clip(inv, -0.3, 0.3)  # Clip returns to avoid extreme values
        return last_price_base * np.exp(np.cumsum(inv))
    else:
        return inv

    