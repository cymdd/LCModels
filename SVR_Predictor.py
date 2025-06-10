import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import os
import glob
import time
from math import sqrt

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    return mse, rmse, mae, corr

start_time = time.time()
folder_path = "your/folder/path" #Please change to your data path
files_number = 100# Please change to your data volume
train_mse_values = []
train_rmse_values = []
train_mae_values = []
train_corr_values = []
test_mse_values = []
test_rmse_values = []
test_mae_values = []
test_corr_values = []

files = glob.glob(os.path.join(folder_path, "*"))
files_to_process = files[:files_number]

for i, file_path in enumerate(files_to_process):
    print(f"Processing file {i+1}/{len(files_to_process)}: {file_path}")
    data = pd.read_csv(file_path)
    data = data["mag"].values

    seq_length = 10
    test_ratio = 0.2  

    X, y = [], []
    for j in range(len(data) - seq_length):
        X.append(data[j:j + seq_length])
        y.append(data[j + seq_length])
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)

    print("Starting model training...")
    svr = SVR(kernel='rbf', C=1.0)
    svr.fit(X_train, y_train)
    print("Model training complete.")

    y_train_pred = svr.predict(X_train)
    y_test_pred = svr.predict(X_test)

    train_mse, train_rmse, train_mae, train_corr = calculate_metrics(y_train, y_train_pred)
    test_mse, test_rmse, test_mae, test_corr = calculate_metrics(y_test, y_test_pred)

    train_mse_values.append(train_mse)
    train_rmse_values.append(train_rmse)
    train_mae_values.append(train_mae)
    train_corr_values.append(train_corr)
    test_mse_values.append(test_mse)
    test_rmse_values.append(test_rmse)
    test_mae_values.append(test_mae)
    test_corr_values.append(test_corr)

    print(f"Metrics for file {i+1}: Train MSE: {train_mse}, Test MSE: {test_mse}")

train_avg_mse = np.mean(train_mse_values)
train_avg_rmse = np.mean(train_rmse_values)
train_avg_mae = np.mean(train_mae_values)
train_avg_corr = np.nanmean(train_corr_values)
test_avg_mse = np.mean(test_mse_values)
test_avg_rmse = np.mean(test_rmse_values)
test_avg_mae = np.mean(test_mae_values)
test_avg_corr = np.nanmean(test_corr_values)

execution_time = time.time() - start_time

print("Train set Average MSE:", train_avg_mse)
print("Train set Average RMSE:", train_avg_rmse)
print("Train set Average MAE:", train_avg_mae)
print("Train set Average CORR:", train_avg_corr)
print("Test set Average MSE:", test_avg_mse)
print("Test set Average RMSE:", test_avg_rmse)
print("Test set Average MAE:", test_avg_mae)
print("Test set Average CORR:", test_avg_corr)
print(f"Program execution time: {execution_time} seconds")
