# src/ml/lstm_forecaster.py
# Trains LSTM for 24h price mean forecast, logs to MLflow for BESS integration.
# Run: python lstm_forecaster.py --data_path /path/to/combined_output_with_features.csv
# For GCP: Set MLFLOW_TRACKING_URI=gs://your-bucket/mlruns; deploy as AI Platform job.

import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pyfunc
import os
import joblib
from typing import Any, Dict, Union
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.tracking import MlflowClient
import json

class PriceDataset(Dataset):
    def __init__(self, data, seq_len=24):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to float32 early
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 24 + 1  # Adjust for target window

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]  # (seq_len,)
        target = self.data[idx+self.seq_len:idx+self.seq_len+24].mean()  # Scalar mean
        return seq, torch.tensor([target], dtype=torch.float32)  # seq (seq_len,), target (1,)

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output: mean price

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])  # Last timestep to mean

class LSTMForecasterWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        scaler_path = context.artifacts["scaler"]
        params_path = context.artifacts.get("params", None)  # Optional during validation
        
        # Default params
        hidden_size = 50
        num_layers = 1
        dropout = 0.2
        
        # Load params from JSON if available (real load)
        if params_path and os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
            hidden_size = params.get("hidden_size", hidden_size)
            num_layers = params.get("num_layers", num_layers)
            dropout = params.get("dropout", dropout)
        
        self.model = PriceLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

    def predict(self, context: Any, model_input: np.ndarray) -> np.ndarray:
        scaled_input = self.scaler.transform(model_input.reshape(-1, 1)).reshape(model_input.shape[0], -1, 1)
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
            output = self.model(input_tensor)
            return self.scaler.inverse_transform(output.numpy())

def evaluate_train(model, scaler, scaled_prices):
    split_idx = int(len(scaled_prices) * 0.8)
    train_data = scaled_prices[:split_idx]
    test_data = scaled_prices[split_idx:]
    test_dataset = PriceDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for seq, target in test_loader:
            seq = seq.unsqueeze(-1)
            output = model(seq)
            predictions.append(output.item())
            actuals.append(target.item())

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).squeeze()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).squeeze()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else 0
    r2 = 1 - (np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2))

    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mape", mape)
    mlflow.log_metric("test_r2", r2)
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Mean Prices')
    plt.plot(predictions, label='Predicted Mean Prices')
    plt.legend()
    plt.title('LSTM Forecast vs Actual (Test Set)')
    plot_path = "forecast_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def train_lstm(data_path, hidden_size=320, num_layers=2, epochs=300, batch_size=32, lr=0.001, dropout=0.3):
    df = pd.read_csv(data_path, sep=';', decimal=',')
    prices = df['Price (EUR)'].values
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).squeeze()
    
    dataset = PriceDataset(scaled_prices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PriceLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("BESS Price Forecaster")
    with mlflow.start_run(run_name="LSTM_Price_Mean"):
        params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
            "dropout": dropout
        }
        mlflow.log_params(params)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for seq, target in loader:
                optimizer.zero_grad()
                seq = seq.unsqueeze(-1)
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        evaluate_train(model, scaler, scaled_prices)
        
        model_path = "lstm_price_model.pth"
        torch.save(model.state_dict(), model_path)
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        # Save params as JSON artifact for wrapper
        params_path = "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        artifacts = {"model": model_path, "scaler": scaler_path, "params": params_path}
        mlflow.pyfunc.log_model(name="model", python_model=LSTMForecasterWrapper(), artifacts=artifacts)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "BESS_Price_Forecaster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output_with_features.csv")
    args = parser.parse_args()
    train_lstm(args.data_path)