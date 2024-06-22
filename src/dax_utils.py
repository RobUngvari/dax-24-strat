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
from sktime.transformations.panel.rocket import Rocket
import calendar
import workalendar
from astral.sun import sun
from astral import LocationInfo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_curve, auc
from sklearn.model_selection import BaseCrossValidator

class Tools:
    @staticmethod
    def analyse_na_value(df, var, target):
        df = df.copy()
        df[var] = np.where(df[var].isnull(), 1, 0)
        tmp = df.groupby(var)[target].agg(['mean', 'std'])

        tmp.plot(kind="barh", y="mean", legend=False,
                xerr="std", title=target, color='green')
        plt.show()

    @staticmethod
    def find_optimal_threshold(y_true, y_prob, func):
        thresholds = np.linspace(0, 1, 1000) 
        max_score = -np.inf
        optimal_threshold = None
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int) 
            score = func(y_true, y_pred) 

            if score > max_score:
                max_score = score
                optimal_threshold = threshold
        
        return optimal_threshold, max_score
    
    @staticmethod
    def performance_report(y_test, y_prob, optimal_threshold):
        # Create a heatmap
        sns.heatmap(confusion_matrix((y_test == 1).astype(int), (y_prob[:,1] > optimal_threshold[0]).astype(int)), 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                )

        # Add labels, title, and display the plot
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        print(classification_report((y_test == 1).astype(int), (y_prob[:,1] > optimal_threshold[0]).astype(int)))

    @staticmethod
    def get_tree_model_feature_importances(model, columns):
        importances = model.feature_importances_
        feature_importance_dict = dict(zip(columns, importances))
        return pd.Series(feature_importance_dict).sort_values(ascending=False)

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
        
        X = X.replace(0, np.nan) # correct strange 0 value days

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
        
    @staticmethod
    def year_days_passed(current_date):
        return (current_date.date() - dt.date(current_date.year, 1, 1)).days
    
    @staticmethod
    def year_days_left(current_date):
        return (dt.date(current_date.year, 12, 31) - current_date.date()).days
    
    @staticmethod
    def month_days_passed(current_date):
        return (current_date.date() - dt.date(current_date.year, current_date.month, 1)).days
    
    @staticmethod
    def month_days_left(current_date):
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        return (current_date.date() - dt.date(current_date.year, current_date.month, last_day)).days

    @staticmethod
    def month_first_day(current_date):
        return calendar.monthrange(current_date.year, current_date.month)[0]
    
    @staticmethod
    def month_length(current_date):
        return calendar.monthrange(current_date.year, current_date.month)[1]
    
    @staticmethod
    def get_last_weekday_of_type(current_date, weekday=calendar.FRIDAY):
        return max(week[weekday] for week in calendar.monthcalendar(current_date.year, current_date.month))
    
    @staticmethod
    def get_sunrise_dusk(current_date):
        CITY = LocationInfo("Frankfurt", "Germany", "Europe/Germany", 50.110924, 8.682127)
        s = sun(CITY.observer, date=current_date)
        sunrise = s['sunrise']
        dusk = s['dusk']
        return (sunrise - dusk).seconds / 3600

    @staticmethod
    def get_number_of_business_days_in_month(current_date):
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        rng = pd.date_range(current_date.replace(day=1), periods=last_day, freq='D')
        return len(pd.bdate_range(rng[0], rng[-1]))
    
    @staticmethod
    def get_number_of_non_business_days(current_date):
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        rng = pd.date_range(current_date.replace(day=1), periods=last_day, freq='D')
        business_days = pd.bdate_range(rng[0], rng[-1])
        return last_day - len(business_days)
        
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        temporal_variable = X.reset_index()[self.temporal_variable_name]
        self.temporal_variable = temporal_variable
        X['year_days_passed'] = self.temporal_variable.apply(lambda x: __class__.year_days_passed(x)).values
        X['year_days_left'] = self.temporal_variable.apply(lambda x: __class__.year_days_left(x)).values
        X['month_days_passed'] = self.temporal_variable.apply(lambda x: __class__.month_days_passed(x)).values
        X['month_days_left'] = self.temporal_variable.apply(lambda x: __class__.month_days_left(x)).values
        X['month_first_day'] = self.temporal_variable.apply(lambda x: __class__.month_first_day(x)).values
        X['month_length'] = self.temporal_variable.apply(lambda x: __class__.month_length(x)).values
        X['last_friday'] = self.temporal_variable.apply(lambda x: __class__.get_last_weekday_of_type(x)).values
        X['sunrise_dusk'] = self.temporal_variable.apply(lambda x: __class__.get_sunrise_dusk(x)).values
        X['business_days_in_month'] = self.temporal_variable.apply(lambda x: __class__.get_number_of_business_days_in_month(x)).values
        X['non_business_days_in_month'] = self.temporal_variable.apply(lambda x: __class__.get_number_of_non_business_days(x)).values
        X['day_of_week'] = temporal_variable.dt.dayofweek.values
        
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
    
class BinaryTarget(BaseEstimator, TransformerMixin):
    def __init__(self, forward_looking_return):
        self.forward_looking_return = forward_looking_return
        
    def fit(self, X):
        return self

    def transform(self, X):
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.forward_looking_return-1)
        X['target'] = X['close_gdaxi'].shift(-1).rolling(window=indexer, min_periods=self.forward_looking_return//2).sum()
        X = X.dropna()
        X['target'] = (X.target > .01).astype(int).values
        return X

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
    
class KalmanMomentum(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        
    @staticmethod
    def kalman(data_series):
        kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

        state_means, _ = kf.filter(data_series)
        return state_means.flatten()
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[f'{variable}_kmomentum'] = (X[variable] - __class__.kalman(X[variable])).replace([np.inf, -np.inf], np.nan).fillna(0)
        return X
    
class Relative_strength(BaseEstimator, TransformerMixin):   
    def __init__(self, instrument):
        self.instrument = instrument
        
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        t = X[[x for x in X.columns if 'close' in x.lower() and self.instrument in x.lower() and 'kmomentum' not in x]]        
        X[f'{self.instrument}_rsi'] = (~t.isna().any(axis=1)).astype(int) * (100 - (100 / (1 + t[t > 0].mean(axis=1) / t[t < 0].mean(axis=1))))
        X[f'{self.instrument}_rsi'] = X[f'{self.instrument}_rsi'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return X
    
class Stochastic_oscillator(BaseEstimator, TransformerMixin):  
    def __init__(self, instrument):
        self.instrument = instrument
        
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        c = X[[x for x in X.columns if 'close' in x.lower() and self.instrument in x and 'kmomentum' not in x]].iloc[:,0]
        h = X[[x for x in X.columns if 'high' in x.lower() and self.instrument in x and 'kmomentum' not in x]].max(axis=1)
        l = X[[x for x in X.columns if 'low' in x.lower() and self.instrument in x and 'kmomentum' not in x]].min(axis=1)     
        X[f'{self.instrument}_stochastic_oscillator'] = (c-l/h-l)*100
        X[f'{self.instrument}_stochastic_oscillator'] = X[f'{self.instrument}_stochastic_oscillator'].replace([np.inf, -np.inf], np.nan).fillna(0)
        return X

class PeriodicCosineTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[f'{variable}_sin'] = np.sin(2*np.pi*X[variable]/(X[variable].nunique()+1))
            X[f'{variable}_cos'] = np.cos(2*np.pi*X[variable]/(X[variable].nunique()+1))
            X = X.drop(variable, axis=1)
        return X
    
class MovingFunctionApplication(BaseEstimator, TransformerMixin):
    def __init__(self, variables, function, label, windows_sizes=[3, 5]):
        self.variables = variables
        self.windows_sizes = windows_sizes
        self.function = function
        self.label = label
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            for ws in self.windows_sizes:
                X[f'{variable}_avg{self.label}{ws}d'] = X[variable].rolling(ws).apply(self.function)
        return X.iloc[max(self.windows_sizes):,:]
    
class Dispersion(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X['market_dispersion'] = X[self.variables].max(axis=1) - X[self.variables].min(axis=1)
        return X
    
class SavGolFilter(BaseEstimator, TransformerMixin):
    def __init__(self, variables, winwow_sizes, poly_orders):
        self.variables = variables
        self.winwow_sizes = winwow_sizes
        self.poly_orders = poly_orders
    
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            for w in self.winwow_sizes:
                for p in self.poly_orders:
                    X[f'savgol_{variable}_w{w}_p{p}'] = X[variable] - savgol_filter(X[variable], w, p)
        return X
    
class AnomalyDetection(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.clf = IsolationForest(n_estimators=100, 
                                 max_samples='auto', 
                                 max_features=1.0, 
                                 bootstrap=False, 
                                 n_jobs=-1, 
                                 random_state=42, 
                                 verbose=0)
        self.clf.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X['anomaly'] = (self.clf.predict(X) == -1).astype(int)
        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.outlier_boundaries_ = {}
        for var in X.columns:
            upper = X[var].mean() + 3*X[var].std()
            lower = X[var].mean() - 3*X[var].std()
            self.outlier_boundaries_[var] = {'lower':lower, 'upper':upper}
        return self

    def transform(self, X):
        X = X.copy()
        for feature in X.columns:
            X[feature] = X[feature].clip(self.outlier_boundaries_[feature]['lower'], 
                                         self.outlier_boundaries_[feature]['upper'])
        return X

class ReasonableTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        split_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            train_start = i * split_size
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = test_start + split_size
            
            if test_end > n_samples:
                test_end = n_samples

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits