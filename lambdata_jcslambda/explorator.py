"""
wrapper class for df_utils
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display

from .df_utils import tvt_split, extract_date_parts
from .df_utils import describe, value_counts
from .df_utils import barplot_feat_by_target_eq_class
from .df_utils import barplots_low_card_feat_by_target_eq_class


class Explorator(object):
    def __init__(self, dataframe, random_state=None):
        self.dataframe = dataframe
        self.random_state = random_state

    # @property
    # def random_state(self):
    #     return self.random_state

    def tvt_split(self, target: str = ''):
        """
        Split this class's dataframe into train, validation, and test sets.

        :param target: name of a column in df which is passed as stratify
            parameter to sklearn.model_selection.train_test_split(), optional
        :param random_state: define the random_state

        :returns: tuple of 3 dataframes - (train, validation, test)
        """
        return tvt_split(self.dataframe, target, self.random_state)

    def extract_date_parts(self, date_column: str, simple=True, inplace=False):
        """
        Convert a column to datetime, then replace it with columns representing
        datetime parts: month, day, year, etc.

        :param date_column: name of date-like column
        :param simple: if True, only convert to year, month, and day
            if False, also include day_of_week, day_of_year, week, and quarter
            optional, default: False

        :returns: a new dataframe with date_column removed and other
            columns added
        """
        df = extract_date_parts(self.dataframe, date_column, simple)
        if inplace:
            self.dataframe = df
            return
        else:
            return df

    def describe(self, formatter={'all': lambda x: f'{x}'}):
        """
        Describe type, total, present, null, nunique, minified_unique,
            and unique items in this class's dataframe. total=number of row.
            present=count of not-nan values. null=count of nan values.
            nunique=count of unique values. minified_unique=count of
            normalized strings. unique=string of unique values.
        If nunique == minified_unique for all columns, minified_unique is
            dropped from the output.

        :param formatter: numpy formatter used in numpy.array2string
            pass None to use default numpy formatting

        :returns: pandas dataframe describing this class's dataframe
        """
        return describe(self.dataframe, formatter)

    def barplot_feat_by_target_eq_class(
        self,
        feature: str,
        target: str,
        target_class,
        ylim=0.7,
        figsize=(9, 6)
    ):
        """
        Display a Seaborn barplot of x feature by (y target == target_class).

        :param feature: name of column to use as x feature
        :param target: name of column to use as y target
        :param target_class: value whose equality is tested with each value in
            y target
        :param ylim: upper limit of y-axis, optional, default: 0.7
        :param figsize: size of matplotlib figure, optional, default: (9, 6)

        :returns: nothing
        """
        barplot_feat_by_target_eq_class(
            feature=feature,
            target=target,
            target_class=target_class,
            dataframe=self.dataframe,
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
        """
        Display Seaborn barplots of x feature by (y target == target_class)
            for features in a dataframe with <= nunique unique values.

        :param target: name of column to use as y target
        :param target_class: value whose equality is tested with each value in
            y target
        :param nunique: limit features to those which have <= this value unique
            values, optional, default: 15
        :param ylim: upper limit of y-axis, optional, default: 0.7
        :param figsize: size of matplotlib figure, optional, default: (9, 6)

        :returns: nothing
        """
        barplots_low_card_feat_by_target_eq_class(
            target=target,
            target_class=target_class,
            dataframe=self.dataframe,
            nunique=nunique,
            ylim=ylim,
            figsize=figsize
        )

    def value_counts(self, features=None):
        """
        Display value counts and normalized value counts together for one or
            more features in this class's dataframe.

        :param features: a feature or list of features found in the dataframe

        :returns: nothing
        """
        if features is None:
            features = self.dataframe.columns.to_list()
        value_counts(self.dataframe, features)
