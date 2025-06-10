import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Updated hyperparameters
num_epochs = 300
learning_rate = 0.01
hidden_size = 20
num_layers = 3
input_size = 1
num_classes = 1
seq_length = 5

dropout_rate = 0.2
folder_path = "your/folder/path" #Please change to your data path
files_number = 100  # Please change to your data volume
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


    
# Convert data into dataset
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
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['mag'].values.reshape(-1,1))
# Split data into training and testing sets
    train_size = int(len(data_scaled) * 0.8)
    test_size = len(data_scaled) - train_size
    train_data = data_scaled[0:train_size,:]
    test_data = data_scaled[train_size:len(data_scaled),:]
    x, y = sliding_windows(data_scaled, seq_length)
    trainX = torch.Tensor(np.array(x[0:train_size]))
    trainY = torch.Tensor(np.array(y[0:train_size]))
    testX = torch.Tensor(np.array(x[train_size:len(x)]))
    testY = torch.Tensor(np.array(y[train_size:len(y)]))

    # Initialize the model
    model = LSTM(num_classes, input_size, hidden_size, num_layers,dropout_rate)
    criterion = torch.nn.MSELoss()    # Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = model(trainX)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        #if (epoch+1) % 100 == 0:
            #print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    # Test the model
    model.eval()
    test_predict = model(testX)
    train_predict = model(trainX)
    
    
    

    train_mse = mean_squared_error(trainY.detach().numpy(), train_predict.detach().numpy())
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(trainY.detach().numpy(), train_predict.detach().numpy())
    
    
    test_mse = mean_squared_error(testY.detach().numpy(), test_predict.detach().numpy())
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(testY.detach().numpy(), test_predict.detach().numpy())
    test_true_values = testY.cpu().detach().numpy()
    test_predicted_values = test_predict.cpu().detach().numpy()
    train_true_values = trainY.cpu().detach().numpy()
    train_predicted_values = train_predict.cpu().detach().numpy()
                                                                                                                                                                       
    
    train_correlation_matrix = np.corrcoef(train_true_values.flatten(), train_predicted_values.flatten())
    test_correlation_matrix = np.corrcoef(test_true_values.flatten(), test_predicted_values.flatten())
    
    test_corr = test_correlation_matrix[0, 1]
    train_corr = train_correlation_matrix[0, 1]
    
    train_mse_values.append(train_mse)
    test_mse_values.append(test_mse)
    train_rmse_values.append(train_rmse)
    test_rmse_values.append(test_rmse)
    train_mae_values.append(train_mae)
    test_mae_values.append(test_mae)
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
