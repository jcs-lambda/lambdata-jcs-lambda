"""
wrapper class for df_utils
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display

from df_utils import tvt_split, extract_date_parts, describe, value_counts
from df_utils import barplot_feat_by_target_eq_class
from df_utils import barplots_low_card_feat_by_target_eq_class


class Explorator(object):
    def __init__(self, dataframe, random_state=None):
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
        return tvt_split(self.dataframe, target, self.random_state)

    def extract_date_parts(self, date_column: str, simple=True, inplace=False):
        df = extract_date_parts(self.dataframe, date_column, simple)
        if inplace:
            self.dataframe = df
            return
        else:
            return df

    def describe(self, formatter={'all': lambda x: f'{x}'}):
        return describe(self.dataframe, formatter)

    def barplot_feat_by_target_eq_class(
        self,
        feature: str,
        target: str,
        target_class,
        ylim=0.7,
        figsize=(9, 6)
    ):
        barplot_feat_by_target_eq_class(
            feature=feature,
            target=target,
            target_class=target_class,
            dataframe=self.dataframe
            ylim=ylim,
            figsize=figsize
        )

    def barplots_low_card_feat_by_target_eq_class(
        self,
        target: str,
        target_class,
        nunique=15,
        ylim=0.7,
        figsize=(9, 6)
    ):
        barplots_low_card_feat_by_target_eq_class(
            target=target,
            target_class=target_class,
            dataframe=self.dataframe,
            nunique=nunique,
            ylim=ylim,
            figsize=figsize
        )

    def value_counts(self, features=None):
        if features is None:
            features = self.dataframe.columns.to_list()
        value_counts(self.dataframe, features)
