"""Wrapper class for df_utils

Explorator: class to operate on a single dataframe with df_utils functions.
"""

from .df_utils import tvt_split, extract_date_parts
from .df_utils import describe, value_counts
from .df_utils import barplot_feat_by_target_eq_class
from .df_utils import barplots_low_card_feat_by_target_eq_class


class Explorator(object):
    """Operate on a dataframe with functions from df_utils module.

    Methods:

    tvt_split: split a dataframe into 3 sets

    expand_date_parts: replace date-like column with date part columns

    describe: create a summary dataframe

    barplot_feat_by_target_eq_class: barplot of feature1 by feature2==X

    barplots_low_card_feat_by_target_eq_class: barplot for each low-cardinality
    feature by some_feature==X

    value_counts: actual and normalized value counts

    Properties:

    df: pandas dataframe

    random_state: int or None

    train: training set, after split

    val: validation set, after split

    test: test set, after split
    """

    def __init__(self, dataframe, random_state=None):
        """Hold a dataframe and operate on it with functions from df_utils.

        Parameters:

        dataframe: pandas dataframe. a copy will be stored as an instance
        variable

        random_state: int or None, optional, default: None.
        """
        self.df = dataframe.copy()
        self.random_state = random_state

    def tvt_split(self, target: str = ''):
        """Split a dataframe into train, validation, and test sets.

        Parameters:

        target: name of a column passed as stratify parameter to
        sklearn.model_selection.train_test_split(), optional

        Returns:

        tuple of 3 dataframes: (train, validation, test)
        """
        return tvt_split(self.df, target, self.random_state)

    def expand_date_parts(self, date_column: str, simple=True, inplace=False):
        """Replace single date-like column with date parts columns.

        Parameters:

        date_column: name of date-like column to send to pandas.to_datetime()

        simple: True - convert to year, month, and day. False - also include
        day_of_week, day_of_year, week, and quarter, optional, default: False

        inplace: True - replace the current dataframe with the expanded one.
        False - return expanded dataframe. optional, default: False

        Returns:

        pandas dataframe with date_column removed and other columns added
        """
        df = expand_date_parts(self.df, date_column, simple)
        if inplace:
            self.df = df
            return
        else:
            return df

    def describe(self, formatter={'all': lambda x: f'{x}'}):
        """Return a dataframe with summary description.

        Parameters:

        formatter: numpy formatter used in numpy.array2string. optional,
        pass None to use default numpy formatting.

        Returns

        pandas dataframe
        """
        return describe(self.df, formatter)

    def barplot_feat_by_target_eq_class(
            self,
            feature: str,
            target: str,
            target_class,
            ylim=0.7,
            figsize=(9, 6)):
        """Display barplot of x feature by (y target == target_class).

        Parameters:

        feature: name of column to use as x feature

        target: name of column to use as y target

        target_class: value tested for equality with each value in y target

        ylim: upper limit of y-axis, optional, default: 0.7

        figsize: size of matplotlib figure, optional, default: (9, 6)

        Returns:

        nothing
        """
        barplot_feat_by_target_eq_class(
            feature=feature,
            target=target,
            target_class=target_class,
            dataframe=self.df,
            ylim=ylim,
            figsize=figsize
        )

    def barplots_low_card_feat_by_target_eq_class(
            self,
            target: str,
            target_class,
            nunique=15,
            ylim=0.7,
            figsize=(9, 6)):
        """Display barplots of x feature by (y target == target_class)
        for features with <= nunique unique values.

        Parameters:

        target: name of column to use as y target

        target_class: value whose equality is tested with each value in target

        nunique: max number of unique values for features to plot

        ylim: upper limit of y-axis, optional, default: 0.7

        figsize: size of matplotlib figure, optional, default: (9, 6)

        Returns:

        nothing
        """
        barplots_low_card_feat_by_target_eq_class(
            target=target,
            target_class=target_class,
            dataframe=self.df,
            nunique=nunique,
            ylim=ylim,
            figsize=figsize
        )

    def value_counts(self, features=None):
        """Display actual and normalized value counts together for one or
        more features.

        Parameters:

        features: a feature or list-like of features

        Returns:

        nothing
        """
        if features is None:
            features = self.df.columns.to_list()
        value_counts(self.df, features)
