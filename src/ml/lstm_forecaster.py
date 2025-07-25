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
from typing import Any, Dict, Union  # For type hints in predict

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
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output: mean price

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last timestep to mean

class LSTMForecasterWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        scaler_path = context.artifacts["scaler"]
        self.model = PriceLSTM()  # Re-instantiate the model class
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))  # Load state_dict safely
        self.model.eval()
        self.scaler = joblib.load(scaler_path)  # Load fitted scaler

    def predict(self, context: Any, model_input: np.ndarray) -> np.ndarray:
        # model_input: seq of prices (batch, seq_len) - assume raw, scale and reshape
        scaled_input = self.scaler.transform(model_input.reshape(-1, 1)).reshape(model_input.shape[0], -1, 1)
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
            output = self.model(input_tensor)
            return self.scaler.inverse_transform(output.numpy())  # Inverse scale output

def train_lstm(data_path, hidden_size=50, num_layers=1, epochs=50, batch_size=32, lr=0.001):
    df = pd.read_csv(data_path, sep=';', decimal=',')
    prices = df['Price (EUR)'].values
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).squeeze()  # (n,)
    
    dataset = PriceDataset(scaled_prices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PriceLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("BESS Price Forecaster")
    with mlflow.start_run(run_name="LSTM_Price_Mean"):
        mlflow.log_params({"hidden_size": hidden_size, "num_layers": num_layers, "epochs": epochs, "lr": lr})
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for seq, target in loader:
                optimizer.zero_grad()
                seq = seq.unsqueeze(-1)  # (batch, seq_len, 1)
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        model_path = "lstm_price_model.pth"
        torch.save(model.state_dict(), model_path)  # Save state_dict only
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        artifacts = {"model": model_path, "scaler": scaler_path}
        mlflow.pyfunc.log_model("model", python_model=LSTMForecasterWrapper(), artifacts=artifacts)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "BESS_Price_Forecaster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output_with_features.csv")
    args = parser.parse_args()
    train_lstm(args.data_path)