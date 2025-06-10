import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import glob
import warnings
import time

warnings.filterwarnings("ignore")

# Function: Find the best ARIMA model using AIC
def best_arima_model(data, max_p=5, max_d=2, max_q=5):
    best_aic = np.inf
    best_order = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    temp_model = ARIMA(data, order=(p, d, q))
                    temp_model_fit = temp_model.fit()
                    if temp_model_fit.aic < best_aic:
                        best_aic = temp_model_fit.aic
                        best_order = (p, d, q)
                except Exception as e:
                    continue

    return best_order

# Folder path (replace with your actual folder path)
folder_path = "your/folder/path" #Please change to your data path
files_number = 100# Please change to your data volume
files = glob.glob(f"{folder_path}/*.csv")  # Assume CSV format

# Initialize lists to hold performance metrics
test_mse_values = []
test_rmse_values = []
test_mae_values = []
test_corr_values = []
train_mse_values = []
train_rmse_values = []
train_mae_values = []
train_corr_values = []

start_time = time.time()

# Process each file 
files_to_process = files[:100]# Please change to your data volume
c=0
for file_path in files_to_process:
    try:
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
        df = df[['mag']].dropna().sort_index()  # Ensure no NaNs and sorted by time

        # Normalize the magnitude values to [0, 1]
        scaler = MinMaxScaler()
        df['scaled_mag'] = scaler.fit_transform(df[['mag']])

        # Split data into training (80%) and testing (20%) sets
        train_size = int(len(df) * 0.8)
        train, test = df['scaled_mag'].iloc[:train_size], df['scaled_mag'].iloc[train_size:]

        # Use the custom function to identify the best ARIMA model parameters
        order = best_arima_model(train)

        # Fit the ARIMA model using the determined best parameters
        model = ARIMA(train, order=order)
        model_fit = model.fit()

        # Forecast for the test set
        predictions_test = model_fit.forecast(steps=len(test))

        # Forecast the training set using in-sample predictions
        predictions_train = model_fit.predict(start=0, end=len(train) - 1)

        # Evaluate predictions for the test set
        test_mse = mean_squared_error(test, predictions_test)
        test_rmse = sqrt(test_mse)
        test_mae = mean_absolute_error(test, predictions_test)
        test_corr = np.corrcoef(test, predictions_test)[0, 1]

        # Evaluate predictions for the training set
        train_mse = mean_squared_error(train, predictions_train)
        train_rmse = sqrt(train_mse)
        train_mae = mean_absolute_error(train, predictions_train)
        train_corr = np.corrcoef(train, predictions_train)[0, 1]

        # Collect results
        test_mse_values.append(test_mse)
        test_rmse_values.append(test_rmse)
        test_mae_values.append(test_mae)
        test_corr_values.append(test_corr)
        train_mse_values.append(train_mse)
        train_rmse_values.append(train_rmse)
        train_mae_values.append(train_mae)
        train_corr_values.append(train_corr)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        continue

end_time = time.time()
execution_time = end_time - start_time

# Calculate and print average performance metrics across all files
test_avg_mse = np.mean(test_mse_values)
test_avg_rmse = np.mean(test_rmse_values)
test_avg_mae = np.mean(test_mae_values)
test_avg_corr = np.mean(test_corr_values)
train_avg_mse = np.mean(train_mse_values)
train_avg_rmse = np.mean(train_rmse_values)
train_avg_mae = np.mean(train_mae_values)
train_avg_corr = np.mean(train_corr_values)


print("Training set Average MSE:", train_avg_mse)
print("Training set Average RMSE:", train_avg_rmse)
print("Training set Average MAE:", train_avg_mae)
print("Training set Average Correlation:", train_avg_corr)
print("Test set Average MSE:", test_avg_mse)
print("Test set Average RMSE:", test_avg_rmse)
print("Test set Average MAE:", test_avg_mae)
print("Test set Average Correlation:", test_avg_corr)
print(f"Total execution time: {execution_time} seconds")
