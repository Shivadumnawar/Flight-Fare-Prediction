# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:28:46 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data= pd.read_excel('Data_Train.xlsx')

train_data.info()

train_data.columns

train_data.describe()

train_data.dropna(inplace=True)

train_data.isnull().sum() # no null values

# creating Journey_day and Journey_month features from Date_of_Journey feature

train_data['Journey_day']= pd.to_datetime(train_data['Date_of_Journey'], format= ('%d/%m/%Y')).dt.day 

train_data['Journey_month']= pd.to_datetime(train_data['Date_of_Journey'], format= ('%d/%m/%Y')).dt.month 

train_data.drop(['Date_of_Journey'], axis=1, inplace= True)

# creating 'Dep_hr' and 'Dep_min' features from 'Dep_Time' feature

train_data['Dep_hr']= pd.to_datetime(train_data['Dep_Time']).dt.hour

train_data['Dep_min']= pd.to_datetime(train_data['Dep_Time']).dt.minute

train_data.drop(['Dep_Time'], axis=1, inplace= True)

# creating 'Arrival_hr' and 'Arrival_min' features from 'Arrival_Time' feature

train_data['Arrival_hr']= pd.to_datetime(train_data['Arrival_Time']).dt.hour

train_data['Arrival_min']= pd.to_datetime(train_data['Arrival_Time']).dt.minute

train_data.drop(['Arrival_Time'], axis=1, inplace= True)


train_data['Duration'].value_counts()

# converted 'Duration' feature's object type to int type

train_data['Duration']= train_data['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m','*1').apply(eval)


sns.catplot(x='Airline', y= 'Price', data= train_data.sort_values('Price', ascending=False), kind='boxen')
plt.xticks(rotation=90)
plt.tight_layout()

# price of Jet Airways Business is significantly high compared to other airlines

''' converting categorical features into numerical variables'''

train_data['Airline'].value_counts()

Airline= pd.get_dummies(train_data['Airline'], drop_first=True, prefix='Airline')

#
train_data['Source'].value_counts()

Source= pd.get_dummies(train_data['Source'], drop_first=True, prefix= 'Source')

#
train_data['Destination'].value_counts()

Destination= pd.get_dummies(train_data['Destination'], drop_first=True , prefix= 'Destination')


train_data.drop(['Airline', 'Destination', 'Source'], axis=1, inplace=True)

train_df= pd.concat([train_data, Airline, Destination, Source], axis=1)
 

train_df['Total_Stops'].value_counts()

# 'Total_Stops' feature is ordinal data type

train_df['Total_Stops'].replace({'non-stop': 0, '1 stop' : 1, 
                                   '2 stops': 2, '3 stops' : 3, '4 stops' : 4}, inplace=True) 

#
train_df['Additional_Info'].value_counts()

# 'Additional_Info'  contains  almost 80% 'No info' 

train_df.drop(['Additional_Info'], axis=1, inplace=True)

#
train_df['Route'].value_counts()

# Route and 'Total_Stops' are related to each other

train_df.drop(['Route'], axis=1, inplace=True)

# correlation

c= train_data.corr()
plt.figure(figsize=(16,12))
sns.heatmap(c, cmap='coolwarm', annot=True)
plt.yticks(rotation=0)

train_df.columns

train_df.isnull().sum()

train_df= train_df[['Duration', 'Total_Stops', 'Journey_day', 'Journey_month',
       'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Price']]

X= train_df.iloc[:, :-1]

y= train_df.iloc[:, -1]

# important features

from sklearn.ensemble import ExtraTreesRegressor

model= ExtraTreesRegressor()
model.fit(X,y)

model.feature_importances_

# visualize important features
feat_importances= pd.Series(model.feature_importances_, index= X.columns)

plt.figure(figsize=(12,10))
feat_importances.nlargest(20).plot(kind='barh')
plt.tight_layout()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=75)


from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor()

rf.fit(X_train, y_train)

y_pred= rf.predict(X_test)

rf.score(X_train, y_train)

rf.score(X_test, y_test)

plt.figure(figsize=(12,8))
sns.distplot(y_test-y_pred)

plt.figure(figsize=(12,8))
plt.scatter(y_test, y_pred)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

from sklearn.metrics import mean_squared_error, mean_absolute_error

mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))

# selecting best hyperparameters

from sklearn.model_selection import RandomizedSearchCV

n_estimators= [ int(x) for x in np.linspace(100,1200,12)]

max_features= ['auto', 'sqrt']

max_depth= [int(x) for x in np.linspace(5,30,6)]

min_samples_split= [2,5,10,15,100]

min_samples_leaf= [1,2,5,10]


random_grid= {
    'n_estimators' : n_estimators,
    'max_features' : max_features,
      'max_depth'  : max_depth,
 'min_samples_split' : min_samples_split,
 'min_samples_leaf' : min_samples_leaf
    }

random_rf= RandomizedSearchCV(estimator= rf, param_distributions= random_grid, 
                              scoring= 'neg_mean_squared_error', cv=5, n_iter=10, verbose=2, n_jobs=1, random_state=75)

random_rf.fit(X_train, y_train)

random_rf.best_params_

random_rf.best_score_

y_pred_test= random_rf.predict(X_test)

r2_score(y_test, y_pred_test)

mean_squared_error(y_test, y_pred_test)
mean_absolute_error(y_test, y_pred_test)
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred_test)))
