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
import mlflow.pytorch
import os

class PriceDataset(Dataset):
    def __init__(self, data, seq_len=24):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len:idx+self.seq_len+24].mean()  # Input seq, target mean of next 24

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
        self.model = torch.load(model_path)
        self.model.eval()
        self.scaler = MinMaxScaler()  # Reload scaler if saved (assume fitted on data)

    def predict(self, context, model_input):
        # model_input: seq of prices (batch, seq_len, 1)
        with torch.no_grad():
            return self.model(torch.tensor(model_input, dtype=torch.float32)).numpy()

def train_lstm(data_path, hidden_size=50, num_layers=1, epochs=50, batch_size=32, lr=0.001):
    df = pd.read_csv(data_path, sep=';', decimal=',')
    prices = df['Price (EUR)'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    dataset = PriceDataset(scaled_prices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PriceLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    mlflow.set_experiment("BESS Price Forecaster")
    with mlflow.start_run(run_name="LSTM_Price_Mean"):
        mlflow.log_params({"hidden_size": hidden_size, "num_layers": num_layers, "epochs": epochs, "lr": lr})
        
        for epoch in range(epochs):
            model.train()
            for seq, target in loader:
                optimizer.zero_grad()
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        model_path = "lstm_price_model.pth"
        torch.save(model, model_path)
        artifacts = {"model": model_path}
        mlflow.pyfunc.log_model("model", python_model=LSTMForecasterWrapper(), artifacts=artifacts)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "BESS_Price_Forecaster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/chavdarbilyanski/energyscale/src/ml/data/combine/combined_output_with_features.csv")
    args = parser.parse_args()
    train_lstm(args.data_path)