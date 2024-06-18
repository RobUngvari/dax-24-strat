import numpy as np
import pandas as pd
import datetime as dt
import re
import os
from sklearn.base import BaseEstimator, TransformerMixin
from pykalman import KalmanFilter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest

class RawDataToReturns(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.loc[:, pd.IndexSlice[['Low', 'High', 'Open','Volume', 'Close'], :]]
        X.columns = ['_'.join(col) for col in X.columns]
        X.columns = [x.replace('^', '').lower() for x in X.columns]
        
        all_business_days = pd.date_range(start=X.index[0], end=X.index[-1], freq='B')
        X = X.reindex(all_business_days)
        
        market_closed = X.isna().any(axis=1)

        X = X.ffill()
        X = np.log(X) - np.log(X.shift(1))#.pct_change()
        
        X['market_closed'] = market_closed.astype(int)
        
        return X.iloc[1:,:]

class BusinessDayLaggedFeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self, variables, days=range(1,4)):
        self.days = days
        self.variables = variables
        
    @staticmethod
    def roll_date(dataframe, day=None, hour=None):
        dataframe = dataframe.reset_index()
        idx_name = dataframe.columns[0]
        # dataframe[idx_name] = dataframe.reset_index()[idx_name] + (dt.timedelta(days=day) if day else dt.timedelta(hours=hour))
        dataframe[idx_name] = dataframe.reset_index()[idx_name] + (BDay(day) if day else dt.timedelta(hours=hour))
        dataframe = dataframe.set_index(idx_name)
        return dataframe
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for day in self.days:
            X = X.join(__class__.roll_date(X[self.variables], day), how='left', rsuffix=f'_day{day}')
        return X.iloc[max(self.days):]
    
class TemporalFeatureEngine(BaseEstimator, TransformerMixin):
    def __init__(self, temporal_variable_name):
        self.temporal_variable_name = temporal_variable_name
        
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        temporal_variable = X.reset_index()[self.temporal_variable_name]
        self.temporal_variable = temporal_variable.values
        X['day_of_month'] = temporal_variable.dt.day.values
        X['day_of_week'] = temporal_variable.dt.dayofweek.values
        X['day_of_year'] = temporal_variable.dt.day_of_year.values
        return X
    
class SimpleMissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan) # quick fix could be problematic
        X = X.ffill()
        return X