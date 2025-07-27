# src/ml/lstm_forecaster.py
# Trains LSTM for full 24h price sequence forecast, logs to MLflow for BESS integration.
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

class PriceDataset(Dataset):
    def __init__(self, data, seq_len=24, forecast_len=24):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to float32 early
        self.seq_len = seq_len
        self.forecast_len = forecast_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.forecast_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]  # (seq_len,)
        target = self.data[idx+self.seq_len:idx+self.seq_len+self.forecast_len]  # (forecast_len,)
        return seq.unsqueeze(-1), target.unsqueeze(-1)  # (seq_len, 1), (forecast_len, 1)

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2, forecast_len=24):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_len)  # Output full 24h sequence

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.fc(out[:, -1, :])  # Last timestep to 24h forecast

class LSTMForecasterWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        scaler_path = context.artifacts["scaler"]
        # New: Load from params artifact if available (preferred)
        hidden_size = 50
        num_layers = 1
        dropout = 0.2
        if "params" in context.artifacts:
            params_path = context.artifacts["params"]
            params = joblib.load(params_path)
            hidden_size = params.get("hidden_size", hidden_size)
            num_layers = params.get("num_layers", num_layers)
            dropout = params.get("dropout", dropout)
        else:
            # Fallback to MLflow run (if active) or defaults
            try:
                client = MlflowClient()
                run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
                if run_id:
                    run = client.get_run(run_id)
                    hidden_size = int(run.data.params.get("hidden_size", hidden_size))
                    num_layers = int(run.data.params.get("num_layers", num_layers))
                    dropout = float(run.data.params.get("dropout", dropout))
            except Exception as e:
                print(f"Warning: Could not fetch params from MLflow (using defaults): {e}")
        
        self.model = PriceLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

    def predict(self, context: Any, model_input: np.ndarray) -> np.ndarray:
        scaled_input = self.scaler.transform(model_input.reshape(-1, 1)).reshape(model_input.shape[0], -1, 1)
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
            output = self.model(input_tensor)
            return self.scaler.inverse_transform(output.numpy())  # (batch, 24)

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
            output = model(seq)
            predictions.append(output.squeeze().numpy())
            actuals.append(target.squeeze().numpy())

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(len(predictions), -1)
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).reshape(len(actuals), -1)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else 0
    r2 = 1 - (np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2))

    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mape", mape)
    mlflow.log_metric("test_r2", r2)
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")

    # Plot sample sequence
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[0], label='Actual Prices (Sample 24h)')
    plt.plot(predictions[0], label='Predicted Prices (Sample 24h)')
    plt.legend()
    plt.title('LSTM Full 24h Forecast vs Actual (Test Sample)')
    plot_path = "forecast_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)

def train_lstm(data_path, hidden_size=350, num_layers=1, epochs=400, batch_size=32, lr=0.003, dropout=0.2):
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
    with mlflow.start_run(run_name="LSTM_Price_Sequence"):
        mlflow.log_params({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
            "dropout": dropout
        })
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for seq, target in loader:
                optimizer.zero_grad()
                output = model(seq)
                loss = criterion(output, target.squeeze(-1))  # Match shapes
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        evaluate_train(model, scaler, scaled_prices)

        params_dict = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout
        }
        params_path = "params.pkl"
        joblib.dump(params_dict, params_path)
        
        model_path = "lstm_price_model.pth"
        torch.save(model.state_dict(), model_path)  # Save state_dict only
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        artifacts = {
            "model": model_path,
            "scaler": scaler_path,
            "params": params_path
            }
        mlflow.pyfunc.log_model(name="model", python_model=LSTMForecasterWrapper(), artifacts=artifacts)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "BESS_Price_Forecaster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output_with_features.csv")
    args = parser.parse_args()
    train_lstm(args.data_path)