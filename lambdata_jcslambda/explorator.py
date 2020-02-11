"""
class of utility functions for exploring a dataframe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display

class Explorator(object):
    def __init__(self, dataframe, random_state=13):
        self.dataframe = dataframe
        self.random_state = random_state
    
    # @property
    # def random_state(self):
    #     return self.random_state

    def tvt_split(self, target: str = ''):
        """
        Split a pandas dataframe into train, validation, and test sets.

        :param df: pandas dataframe, required
        :param target: name of a column in df which is passed as stratify 
            parameter to sklearn.model_selection.train_test_split(), optional
        :param random_state: define the random_state
        :returns: tuple of 3 dataframes - (train, validation, test)
        """
        if target != '' and target in self.dataframe.columns:
            stratify = self.dataframe[target]
        else:
            stratify = None

        train, test = train_test_split(
            self.dataframe,
            test_size=0.2,
            stratify=stratify,
            random_state=self.random_state
        )

        if stratify is not None:
            stratify = train[target]

        train, val = train_test_split(
            train,
            test_size=0.2,
            stratify=stratify,
            random_state=self.random_state
        )

        return train, val, test

    def extract_date_parts(self, date_column: str, simple=True, inplace=False):
        assert date_column in self.dataframe.columns, \
            f'{date_column} not found in dataframe'
        df = self.dataframe.copy()
        datetimes = pd.to_datetime(
            df[date_column], infer_datetime_format=True, errors='coerce')
        df['year'] = datetimes.dt.year
        df['month'] = datetimes.dt.month
        df['day'] = datetimes.dt.day
        if not simple:
            df['day_of_week'] = datetimes.dt.dayofweek
            df['day_of_year'] = datetimes.dt.dayofyear
            df['week'] = datetimes.dt.week
            df['quarter'] = datetimes.dt.quarter
        df.drop(columns=date_column, inplace=True)
        if inplace:
            self.dataframe = df
            return
        else:
            return df

    def describe(self, formatter={'all': lambda x: f'{x}'}):
        def len_minified(series):
            if not series.dtype == 'O':
                return series.nunique()
            return series.fillna('').str.lower().str.replace('[^a-z0-9]', '').nunique()

        data = pd.DataFrame({
            'type': self.dataframe.dtypes,
            'total': self.dataframe.count() + self.dataframe.isnull().sum(),
            'present': self.dataframe.count(),
            'null': self.dataframe.isnull().sum(),
            'nunique': self.dataframe.nunique(),
            'minified_nunique': [len_minified(pd.Series(self.dataframe[column].unique())) for column in self.dataframe.columns],
            'unique': [np.array2string(self.dataframe[column].unique(), separator=', ', formatter=formatter)[1:-1] for column in self.dataframe.columns]
        })

        if (data['nunique'] == data['minified_nunique']).all():
            data.drop(columns='minified_nunique', inplace=True)
        return data

    def barplot_feat_by_target_eq_class(self, feature: str, target: str, target_class, ylim=0.7, figsize=(9, 6)):
        assert feature in self.dataframe.columns, \
            f'FEATURE: {feature} NOT FOUND IN DATAFRAME.columns'
        assert target in self.dataframe.columns, \
            f'TARGET: {target} NOT FOUND IN DATAFRAME.columns'
        assert target_class in self.dataframe[target], \
            f'TARGET CLASS: {target_class} NOT FOUND IN DATAFRAME[\'{target}\']'
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            ax=ax,
            x=self.dataframe[feature],
            y=self.dataframe[target] == target_class
        )
        plt.axhline(y=self.dataframe[target].value_counts(normalize=True)[target_class], color='red')
        plt.title(f'{feature} by % {target}=={target_class}')
        plt.ylim(top=ylim)
        plt.show()

    def barplots_low_card_feat_by_target_eq_class(self, target: str, target_class, nunique=15, ylim=0.7, figsize=(9, 6)):
        assert target in self.dataframe.columns, \
            f'TARGET: {target} NOT FOUND IN DATAFRAME.columns'
        for feature in self.dataframe.columns[(self.dataframe.nunique() <= nunique) & (self.dataframe.nunique() > 1)].drop([target], errors='ignore'):
            self.barplot_feat_by_target_eq_class(feature, target, target_class, ylim=ylim, figsize=figsize)

    def value_counts(self, features=None):
        if features is None:
            features = self.dataframe.columns.to_list()
        if not isinstance(features, (list, set, tuple)):
            features = [features]
        for feature in features:
            df = pd.DataFrame({
                'count': self.dataframe[feature].value_counts().sort_index(),
                'percentage': self.dataframe[feature].value_counts(normalize=True).sort_index()
            })
            df.index.name = feature
            display(df.sort_values(by='count', ascending=False))
        return
