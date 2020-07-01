#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:32:16 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read csv files
station = pd.read_csv("./data/station_data.csv")
station.head(5)

trip = pd.read_csv("./data/trip_data.csv", parse_dates=['Start Date','End Date'])
trip.head(5)

weather = pd.read_csv("./data/weather_data.csv", parse_dates=['Date'])
weather.head(5)

# Update trips' Start and End Station id
trip['Start Station'].value_counts()
trip.loc[trip['Start Station'] == 23, 'Start Station'] = 85 ## Now #85 counts is 127
trip.loc[trip['Start Station'] == 25, 'Start Station'] = 86
trip.loc[trip['Start Station'] == 49, 'Start Station'] = 87
trip.loc[trip['Start Station'] == 69, 'Start Station'] = 88
trip.loc[trip['Start Station'] == 72, 'Start Station'] = 89
trip.loc[trip['Start Station'] == 89, 'Start Station'] = 90

# Check Updated Station to see if they are same
station_1 = station[station.Id.isin([23,25,49,69,72])]
station_2 = station[station.Id.isin([85,86,87,88,89,90])]
## Yes, they are same

# Drop old stations
station_new = station[~station.Id.isin([23,25,49,69,72,89])]

# Create start and end station summary from trip dataset
trip_s = trip.groupby(['Start Station', pd.Grouper(key='Start Date', freq='H')])['Trip ID'].count().reset_index()
trip_s = trip_s.rename(columns = {'Trip ID': 'Start_Counts'})
trip_e = trip.groupby(['End Station', pd.Grouper(key='End Date', freq='H')])['Trip ID'].count().reset_index()
trip_e = trip_e.rename(columns = {'Trip ID': 'End_Counts'})

# Outer join 2 summary tables
trip_m = trip_s.merge(trip_e, how = 'outer', left_on = ['Start Station', 'Start Date'], right_on = ['End Station', 'End Date'])
# Drop End Station and End Date columns
trip_m.drop(columns=['End Station', 'End Date'], inplace=True)
# Fill in missing values with 0 and calculate Net Rate
trip_m['End_Counts'] = trip_m['End_Counts'].fillna(0)
trip_m['Start_Counts'] = trip_m['Start_Counts'].fillna(0)
trip_m['Net Rate']  = trip_m['End_Counts'] - trip_m['Start_Counts']

# Update city with zip in station dataset
station_new.loc[station_new['City'] == 'San Francisco', 'Zip'] = 94107
station_new.loc[station_new['City'] == 'Redwood City', 'Zip'] = 94063
station_new.loc[station_new['City'] == 'Palo Alto', 'Zip'] = 94301
station_new.loc[station_new['City'] == 'Mountain View', 'Zip'] = 94041
station_new.loc[station_new['City'] == 'San Jose', 'Zip'] = 95113

# Define some functions for EDA
def print_dataframe_description(df, col):
    print('Column Name:', col)
    print('Number of Rows:', len(df.index))
    print('Number of Missing Values:', df[col].isnull().sum())
    print('Percent Missing:', df[col].isnull().sum()/len(df.index)*100, '%')
    print('Number of Unique Values:', len(df[col].unique()))
    print('\n')
    
# Explore weather dataset
for col in list(weather.columns.values):
    print_dataframe_description(weather, col)
# Events variable has 84% missing
weather['Events'].value_counts()

# Create dummy variables for Events and Zip variables
event_temp = pd.get_dummies(weather['Events'], prefix='Events', dummy_na=True)
weather_event_temp = weather.merge(event_temp, how = 'left', left_index = True, right_index = True)

zip_temp = pd.get_dummies(weather['Zip'], prefix='Zip', dummy_na=True)
weather_event_zip = weather_event_temp.merge(zip_temp, how = 'left', left_index = True, right_index = True)

# Station left join with weather dataset
station_new_weather_event_zip = weather_event_zip.merge(station_new, how = 'left', on = 'Zip')

# Create pure date column in trip_m dataset
trip_m['Date'] = trip_m['Start Date'].dt.date
station_new_weather_event_zip['Date'] = station_new_weather_event_zip['Date'].dt.date
trip_m = trip_m.rename(columns = {'Start Station': 'Id'})

# Merge two datasets
station_new_weather_event_zip_trip_m = trip_m.merge(station_new_weather_event_zip, on = ['Date','Id'])

# Remove redudant variables and set Id, Start Date to be index
final = station_new_weather_event_zip_trip_m.drop(columns=['Date', 'Events', 'Zip', 'Name', 'City'])
final = final.set_index(['Id', 'Start Date'])


# Split the training, test data
from sklearn.model_selection import train_test_split
target = 'Net Rate'
X = final.loc[:, final.columns != target]
y = final.loc[:, final.columns == target]

for col in list(final.columns.values):
    print_dataframe_description(final, col)
# No variable percent missing above 4%
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train.head(10)

###############################################################
########################### XGBoost ###########################
###############################################################

import xgboost as xgb
from sklearn.metrics import mean_squared_error

model_xg = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

model_xg.fit(X_train, y_train)
preds_xg = model_xg.predict(X_test)

# Calculate the rmse
rmse_xg = np.sqrt(mean_squared_error(y_test, preds_xg))
print("RMSE: %f" % (rmse_xg))
# RMSE: 1.388632

# Plot feature importance
xgb.plot_importance(model_xg)
plt.rcParams['figure.figsize'] = [10, 40]
plt.show()
# End_Counts and Start_Counts variables are two most important variables

#########################################################################
########################### XGBoost - Updated ###########################
#########################################################################
# Updated new X and Y with removing the End_Counts and Start_Counts since the target vairble is the product of these two
X_new = X.copy()
X_new = X_new.drop(columns=['Start_Counts', 'End_Counts'])
y_new = y.copy()
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=123)
X_train_new.head(10)

model_xg_new = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

model_xg_new.fit(X_train_new, y_train_new)
preds_xg_new = model_xg_new.predict(X_test_new)

# Calculate the rmse
rmse_xg_new = np.sqrt(mean_squared_error(y_test_new, preds_xg_new))
print("RMSE: %f" % (rmse_xg_new))
# RMSE: 3.189381

# Plot feature importance
xgb.plot_importance(model_xg_new)
plt.rcParams['figure.figsize'] = [10, 40]
plt.show()
# This time latitude and longitude become the top two important varibales, which makes sense because the geographical location greatly differentiate each BikeShare station

##############################################################################
########################### Potential Improvements ###########################
##############################################################################
# 1. Control coding environment so that others can run the whole code easily without worrying about something like Python package conflict
# 2. Tune machine learning models to improve the accuracy
# 3. Incorporate "Subscription Type" variable in the model so that can add an extra layer of customer information

##############################################################################
########################### Conclusions ######################################
##############################################################################
# From the XGBoost model, it seems BikeShare station's latitude and longitude information are top two important varibales
# and the model's RMSE is 3.189381.
# BeachBoys BikeShare company may do more investigation and pay more attention on each Station since their geographical information and weather conditions have great impact on the intensity use of each station.
