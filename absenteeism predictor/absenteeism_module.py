#!/usr/bin/env python
# coding: utf-8

# In[49]:


# import all the libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.columns = columns
        self.mean = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler = StandardScaler(copy=self.copy,with_mean=self.with_mean,with_std=self.with_std)
        self.scaler.fit(X[self.columns],y)
        self.mean = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled], axis=1)[init_col_order]
    
# Class used for new data predictions
class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        # read the model and scaler
        with open('model','rb') as model_file, open('scaler','rb') as scaler_file:
            self.model = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
    
    # Take the new data file (*.csv) and preprocess it
    def load_and_clean_data(self, data_file):
         # load the data
        df = pd.read_csv(data_file, delimiter=',')
        # store the data in a new variable for use
        self.raw_data = df.copy()
        
        # drop the feature 'ID'
        df = df.drop(['ID'], axis=1)
        # add the feature 'Absenteeism Time in Hours' to preserve the code
        df['Absenteeism Time in Hours'] = 'NaN'
        
        # create a separate dataframe for the dummy variable for each reason for absence
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        # group the reasons into for different types
        def check_if_empty(reason_values):
            if reason_values.isnull().values.any():
                return reason_values.apply(lambda x: 0)
            return reason_values
        reason_type_1 = check_if_empty(reason_columns.loc[:,1:14].max(axis=1))
        reason_type_2 = check_if_empty(reason_columns.loc[:,15:17].max(axis=1))
        reason_type_3 = check_if_empty(reason_columns.loc[:,18:21].max(axis=1))
        reason_type_4 = check_if_empty(reason_columns.loc[:,22:].max(axis=1))
        # concatenate the df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1,reason_type_2,reason_type_3,reason_type_4], axis=1)
        # drop the feature 'Reason for Absence' drom the dataframe to avoid multicolinearity
        df = df.drop(['Reason for Absence'], axis = 1)
        # rename the columns for each type of reason for absence
        columns_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 
                 'Reason type 1', 'Reason type 2', 'Reason type 3', 'Reason type 4']
        df.columns = columns_names
        # reorder the columns in the df
        columns_reordered =  ['Reason type 1','Reason type 2', 'Reason type 3', 'Reason type 4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[columns_reordered]
        
        # convert the type of values from the feature 'Date' into datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        # extract a list of months from the values of the feature 'Date' and create a new feature
        df['Month Value'] = [date.month for date in df['Date']]
        # extract the weekday and create a new feature
        df['Day of the Week'] = df['Date'].apply(lambda date: date.weekday())
        # drop the feature 'Date' drom the dataframe to avoid multicolinearity
        df = df.drop(['Date'],axis=1)
        
        # map the values for education >=2 into a single category because 
        # there're very few observetions for these categories
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # drop unimportant features and the target
        df = df.drop(['Absenteeism Time in Hours','Distance to Work',
           'Daily Work Load Average',
           'Day of the Week'], axis=1)
        
        # if it's necessary to get the preprocessed data, the following line of code can be used
        self.preprocessed_data = df.copy()
        
        # the next line is for the following functions
        self.data = self.scaler.transform(df)
    
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if(self.data is not None):
            pred = self.model.predict_proba(self.data)[:,1]
            return pred
    
    # a function which outputs 0 or 1 based on the model
    def predicted_output_category(self):
        if(self.data is not None):
            pred_outputs = self.model.predict(self.data)
            return pred_outputs 
    
    # predict the outputs & probabilities and add columns with these values at the end of the new data
    def predicted_outputs(self):
        if(self.data is not None):
            self.preprocessed_data['Probability'] = self.model.predict_proba(self.data)[:,-1]
            self.preprocessed_data['Prediction'] = self.model.predict(self.data)
            return self.preprocessed_data
#     def cleaned_data(self):
#         return self.data


# In[50]:


# model = absenteeism_model('model','scaler')
# model.load_and_clean_data('Absenteeism_new_data.csv')
# model.cleaned_data()


# In[ ]:




