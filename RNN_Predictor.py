import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
import glob


num_epochs = 300
learning_rate = 0.01
hidden_size = 20
num_layers = 3
input_size = 1
num_classes = 1
seq_length = 5

dropout_rate = 0.2
folder_path = "your/folder/path" #Please change to your data path
files_number = 100# Please change to your data volume
class RNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.0):
        super(RNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize the hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out)  # Applying dropout
        out = out[:, -1, :]
        out = self.fc(out)
        return out



def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x),np.array(y)

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

files_to_process = files[:files_number] 
for file_path in files_to_process:
    data = pd.read_csv(file_path)
    if data.empty or len(data) < seq_length + 2: # Check if the data is sufficient
        print(f"Skipping file {file_path} because of insufficient data。")
        continue
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['mag'].values.reshape(-1,1))

    
    x, y = sliding_windows(data_scaled, seq_length)
    train_size = int(len(x) * 0.8)
    test_size = len(x) - train_size

    if train_size == 0 or test_size == 0: #Check if the dataset is valid
        print(f"Skipping file {file_path} because the training set or test set is empty.")
        continue

    trainX = torch.Tensor(x[0:train_size])
    trainY = torch.Tensor(y[0:train_size])
    testX = torch.Tensor(x[train_size:len(x)])
    testY = torch.Tensor(y[train_size:len(y)])

    
    model = RNN(num_classes, input_size, hidden_size, num_layers, dropout=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(trainX)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()

    
    model.eval()
    with torch.no_grad(): 
        test_predict = model(testX)
        train_predict = model(trainX)

        train_mse = mean_squared_error(trainY.numpy(), train_predict.numpy())
        test_mse = mean_squared_error(testY.numpy(), test_predict.numpy())

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_mae = mean_absolute_error(trainY.numpy(), train_predict.numpy())
        test_mae = mean_absolute_error(testY.numpy(), test_predict.numpy())

        # The corr calculation may need to handle the case where there are too few data points
        if len(trainY) > 1 and len(testY) > 1:
            train_corr = np.corrcoef(trainY.numpy().flatten(), train_predict.numpy().flatten())[0, 1]
            test_corr = np.corrcoef(testY.numpy().flatten(), test_predict.numpy().flatten())[0, 1]
        else:
            train_corr = test_corr = None 

        
        train_mse_values.append(train_mse)
        test_mse_values.append(test_mse)
        train_rmse_values.append(train_rmse)
        test_rmse_values.append(test_rmse)
        train_mae_values.append(train_mae)
        test_mae_values.append(test_mae)
        if train_corr is not None and test_corr is not None:
            train_corr_values.append(train_corr)
            test_corr_values.append(test_corr)

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
