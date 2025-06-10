import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os
import glob
import time
import matplotlib.pyplot as plt

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2)
        self.flatten = nn.Flatten()
        
        conv_output_size = 128 * (sequence_length - 3)  
        self.fc1 = nn.Linear(conv_output_size, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        #print(f"Epoch {epoch+1}/{epochs} completed.")


def predict(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            
            predictions.extend(output.cpu().squeeze().tolist() if output.numel() > 1 else [output.item()])
            actuals.extend(y_batch.tolist())
    return actuals, predictions



def evaluate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-10))) * 100
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else None
    return mse, rmse, mae, mape, corr



batch_size = 16
sequence_length = 5
folder_path = "your/folder/path" #Please change to your data path
files_number = 100 # Please change to your data volume
epochs_number = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()


files = glob.glob(os.path.join(folder_path, "*"))
train_mse_values = []
test_mse_values = []
train_rmse_values = []
test_rmse_values = []
train_mae_values = []
test_mae_values = []
train_corr_values = []
test_corr_values = []

files_to_process = files[:files_number] # Please change to your data volume


for file_path in files_to_process:

    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data['normalized_value'] = scaler.fit_transform(data[['mag']])



    
    sequences = create_sequences(data['normalized_value'].values, sequence_length)

    train_val_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    train_sequences, val_sequences = train_test_split(train_val_sequences, test_size=0.25, random_state=42)

    X_train, y_train = train_sequences[:, :-1], train_sequences[:, -1]
    X_val, y_val = val_sequences[:, :-1], val_sequences[:, -1]
    X_test, y_test = test_sequences[:, :-1], test_sequences[:, -1]

    X_train_t = torch.tensor(X_train).float().unsqueeze(1)
    X_val_t = torch.tensor(X_val).float().unsqueeze(1)
    X_test_t = torch.tensor(X_test).float().unsqueeze(1)
    y_train_t = torch.tensor(y_train).float()
    y_val_t = torch.tensor(y_val).float()
    y_test_t = torch.tensor(y_test).float()


    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    model = CNN()
    model.to(device)

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    train_model(model, train_loader, criterion, optimizer, epochs=epochs_number, device=device)

    
    actuals_train, predictions_train = predict(model, train_loader,device)
    actuals_test, predictions_test = predict(model, test_loader,device)

    mse_train, rmse_train, mae_train, mape_train, corr_train = evaluate_performance(actuals_train, predictions_train)
    mse_test, rmse_test, mae_test, mape_test, corr_test = evaluate_performance(actuals_test, predictions_test)

    
    train_mse_values.append(mse_train)
    train_rmse_values.append(rmse_train)
    train_mae_values.append(mae_train)
    train_corr_values.append(corr_train)
    test_mse_values.append(mse_test)
    test_rmse_values.append(rmse_test)
    test_mae_values.append(mae_test)
    test_corr_values.append(corr_test)
    
end_time = time.time()
execution_time = end_time - start_time

train_avg_mse = sum(train_mse_values) / len(train_mse_values)
test_avg_mse = sum(test_mse_values) / len(test_mse_values)

train_avg_rmse = sum(train_rmse_values) / len(train_rmse_values)
test_avg_rmse = sum(test_rmse_values) / len(test_rmse_values)

train_avg_mae = sum(train_mae_values) / len(train_mae_values)
test_avg_mae = sum(test_mae_values) / len(test_mae_values)
train_avg_corr = sum(train_corr_values) / len(train_corr_values)
test_avg_corr = sum(test_corr_values) / len(test_corr_values)
print("Train set Average MSE:", train_avg_mse)
print("Test set Average MSE:", test_avg_mse)
print("Train set Average RMSE:", train_avg_rmse)
print("Test set Average RMSE:", test_avg_rmse)
print("Train set Average MAE:", train_avg_mae)
print("Test set Average MAE:", test_avg_mae)
print("Train set Average CORR:", train_avg_corr)
print("Test set Average CORR:", test_avg_corr)
print(f"Program execution time: {execution_time} s")
