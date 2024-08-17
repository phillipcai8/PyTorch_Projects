# Setup - import basic data libraries
#import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#torch.manual_seed(42)

# Read the charging and traffic datasets.
ev_charging_reports = pd.read_csv('datasets/EV charging reports.csv', dtype={'Start_plugin_hour': 'object'}, sep=";")
traffic_reports = pd.read_csv('datasets/Local traffic distribution.csv', sep=";")

# Replace invalid characters with 0.
traffic_reports = traffic_reports.replace('-', '0')

# Merge the datasets together and drop the features that (probably) won't help
ev_charging_traffic = traffic_reports.merge(ev_charging_reports, left_on=['Date_from'], right_on=['Start_plugin_hour'])
ev_charging_traffic = ev_charging_traffic.drop(labels=['session_ID', 'Garage_ID', 'User_ID', 'Shared_ID',
                                                       'Plugin_category', 'Duration_category',
                                                       'Start_plugin', 'Start_plugin_hour', 'End_plugout',
                                                       'End_plugout_hour',
                                                       'Date_from', 'Date_to'], axis=1)
num_features = 9

# Create the relevant mapping dictionaries
user_mapping = {
    'Private': 1,
    'Shared': 2
}
day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
month_mapping = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

# Map the various string column values to their respective integers
ev_charging_traffic['User_type'] = ev_charging_traffic['User_type'].map(user_mapping)
ev_charging_traffic['weekdays_plugin'] = ev_charging_traffic['weekdays_plugin'].map(day_mapping)
ev_charging_traffic['month_plugin'] = ev_charging_traffic['month_plugin'].map(month_mapping)

# Since this data is European, replace instances of ',' with '.' for decimals
ev_charging_traffic['El_kWh'] = ev_charging_traffic['El_kWh'].str.replace(',', '.')
ev_charging_traffic['Duration_hours'] = ev_charging_traffic['Duration_hours'].str.replace(',', '.')

# Convert the types to numeric and replace the NaN values.
ev_charging_traffic = ev_charging_traffic.astype(float)
ev_charging_traffic.fillna(0.0, inplace=True)

# Create the train and test datasets using the overall dataframe
X = ev_charging_traffic.drop(labels='El_kWh', axis=1)
y = ev_charging_traffic['El_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80,
                                                    test_size=0.20,
                                                    random_state=2)

# Train and test linear model
print("\nLinear test=============================>")
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

linear_predictions = linearModel.predict(X_test)
test_mse = mean_squared_error(y_test, linear_predictions)
print(f'Linear MSE is: {test_mse}')
print(f'Linear Root MSE is: {test_mse ** (1/2)}')

# Instantiate the neural network
model = nn.Sequential(
    nn.Linear(num_features, 56),
    nn.ReLU(),
    nn.Linear(56, 28),
    nn.ReLU(),
    nn.Linear(28, 1)
)

# Create the datasets tensors for the NN
X_train = torch.tensor(X_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float).unsqueeze(1)

# Initialize the loss and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.007)

# Check for NaN values
if torch.isnan(X_train).any():
    X_train = torch.where(torch.isnan(X_train), torch.tensor(0.0), X_train)

print("\nX_train=============================>")
num_epochs = 10000
for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

print("\nX_test=============================>")
# Check for NaN values
if torch.isnan(X_test).any():
    X_test = torch.where(torch.isnan(X_test), torch.tensor(0.0), X_test)

# Test the NN
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_MSE = loss(predictions, y_test)

print('Test MSE is ' + str(test_MSE.item()))
print('Test Root MSE is ' + str(test_MSE.item() ** (1 / 2)))
