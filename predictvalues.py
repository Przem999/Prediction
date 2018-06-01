# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:31:30 2018

@author: P
"""

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split



class predictvalues: 
    '''This class help us to predict values by means of regression. We can do it when we have historical data in MS Excel and planned some of those data too in MS Excel.'''
    def __init__(self, historical_data, future_data, wanted_item): 
        '''Initiate prediction with some arguments.
        historical_data is a path with Excel file name. This file contains historical data.
        future_data is a path with Excel file name. This file contains planned future values.
        wanted_item is name of column with apostrophe in historical_data, where are values, which we try to predict.
        number_of_col is number of column in file with historical data, where are values, which we try to predict.'''
        self.historical_data = pd.read_excel(historical_data)
        self.future_data = pd.read_excel(future_data)
        self.wanted_item = wanted_item
    
    def param_materiality(self):
        '''Some of parameters in historical data do not affect predictive values. In param_materiality we use Lasso Regression to choose important parameters from historical data.''' 

        y = self.historical_data[self.wanted_item].values
        X = self.historical_data.drop(self.wanted_item, axis=1).values
        names = self.historical_data.drop(self.wanted_item, axis=1).columns

        # Instantiate a lasso regressor: lasso
        lasso = Lasso(alpha=0.4, normalize=True)
        # Fit the regressor to the data
        lasso.fit(X, y)
        # Compute and print the coefficients
        lasso_coef = lasso.coef_
        print(lasso_coef)

        # Plot the coefficients
        plt.plot(range(len(names)), lasso_coef)
        plt.xticks(range(len(names)), names.values, rotation=60)
        plt.margins(0.02)
        plt.show()

        coef_df = pd.Series(lasso_coef, index=names)

        print('Now you have possibiity to choose the most important columns with data in your file with historical data. In the table below you have your columns names with numbers. The most important are those, which have the highest abosolute value. Those, which have value about zero are not important and you could delete them from you file with historical data in order to calculate prediction.')
        print('**************************')
        print('But remember please: In the file with future data you have to put only those columns, which you have in historical data file')
        print('**************************')
        print(coef_df)
        
    def prediction(self, path_to_future):
        '''We can do a prediction by means of Linear Regression.
        path_to_future :) is path to place, where you want to get predicted values in flat file.'''
        
        def val_or_zero(v):
            return max(0,v[0])
    
        # Create arrays for features and target variable
        y = self.historical_data[self.wanted_item].values
        X = self.historical_data.drop(self.wanted_item, axis=1).values
        # Reshape X and y
        y = y.reshape(-1, 1)
        X = X.reshape(-1, len(self.historical_data.columns)-1)
        # Create training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        # Create the regressor: reg_all
        reg_all = LinearRegression()
        # Fit the regressor to the training data
        reg_all.fit(X_train, y_train)
        # Prediction ********
        
        # Create arrays for features and target variable

        Z = self.future_data.values
        # Reshape Z
        Z = Z.reshape(-1, len(self.future_data.columns))
        zpred_df = pd.DataFrame(reg_all.predict(Z))
        zpred_df[0]=zpred_df.apply(val_or_zero, axis=1)
        zpred_df.columns = [self.wanted_item]
        prediction_df = pd.merge(self.future_data, zpred_df, left_index=True, right_index=True, how='left')
        prediction_df.to_csv(path_to_future)
        print(prediction_df) 







