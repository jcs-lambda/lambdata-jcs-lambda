"""
utility functions for working with dataframes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display


def tvt_split(df, target: str='', random_state=None):
    """
    Split a pandas dataframe into train, validation, and test sets.

    :param df: pandas dataframe, required
    :param target: name of a column in df which is passed as stratify
        parameter to sklearn.model_selection.train_test_split(), optional
    :param random_state: define the random_state

    :returns: tuple of 3 dataframes - (train, validation, test)
    """
    if target != '' and target in df.columns:
        stratify = df[target]
    else:
        stratify = None

    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=stratify,
        random_state=random_state
    )

    if stratify is not None:
        stratify = train[target]

    train, val = train_test_split(
        train,
        test_size=0.2,
        stratify=stratify,
        random_state=random_state
    )

    return train, val, test


def extract_date_parts(dataframe, date_column, simple=True):
    """
    Convert a column to datetime, then replace it with columns representing
    datetime parts: month, day, year, etc.

    :param dataframe: pandas dataframe that contains a date-like column
    :param date_column: name of date-like column
    :param simple: if True, only convert to year, month, and day
        if False, also include day_of_week, day_of_year, week, and quarter
        optional, default: False

    :returns: a new dataframe with date_column removed and other columns added
    """
    assert date_column in dataframe.columns, \
        f'{date_column} not found in dataframe'
    df = dataframe.copy()
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
    return df


def describe(dataframe, formatter={'all': lambda x: f'{x}'}):
    """
    Describe type, total, present, null, nunique, minified_unique, and unique
        items in a dataframe. total=number of row. present=count of not-nan
        values. null=count of nan values. nunique=count of unique values.
        minified_unique=count of normalized strings. unique=string of unique
        values.
    If nunique == minified_unique for all columns, minified_unique is dropped
        from the output.

    :param dataframe: pandas dataframe
    :param formatter: numpy formatter used in numpy.array2string
        pass None to use default numpy formatting

    :returns: pandas dataframe describing the dataframe passed as first param
    """
    def len_minified(series):
        if not series.dtype == 'O':
            return series.nunique()
        return series.fillna('').str.lower().str.replace('[^a-z0-9]', '') \
            .nunique()

    data = pd.DataFrame({
        'type': dataframe.dtypes,
        'total': dataframe.count() + dataframe.isnull().sum(),
        'present': dataframe.count(),
        'null': dataframe.isnull().sum(),
        'nunique': dataframe.nunique(),
        'minified_nunique': [len_minified(
            pd.Series(dataframe[column].unique()))
                             for column in dataframe.columns],
        'unique': [np.array2string(
                    dataframe[column].unique(),
                    separator=', ',
                    formatter=formatter)[1:-1]
                   for column in dataframe.columns]
    })

    if (data['nunique'] == data['minified_nunique']).all():
        data.drop(columns='minified_nunique', inplace=True)
    return data


def barplot_feat_by_target_eq_class(
        feature: str,
        target: str,
        target_class,
        dataframe,
        ylim=0.7,
        figsize=(9, 6)
):
    """
    Display a Seaborn barplot of x feature by (y target == target_class).

    :param feature: name of column to use as x feature
    :param target: name of column to use as y target
    :param target_class: value whose equality is tested with each value in
        y target
    :param dataframe: pandas dataframe containing feature and target columns
    :param ylim: upper limit of y-axis, optional, default: 0.7
    :param figsize: size of matplotlib figure, optional, default: (9, 6)

    :returns: nothing
    """

    assert feature in dataframe.columns, \
        f'FEATURE: {feature} NOT FOUND IN DATAFRAME.columns'
    assert target in dataframe.columns, \
        f'TARGET: {target} NOT FOUND IN DATAFRAME.columns'
    assert target_class in dataframe[target].unique(), \
        f'TARGET CLASS: {target_class} NOT FOUND IN DATAFRAME[\'{target}\']'
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        ax=ax,
        x=dataframe[feature],
        y=dataframe[target] == target_class
    )
    plt.axhline(y=dataframe[target].value_counts(
        normalize=True)[target_class], color='red')
    plt.title(f'{feature} by % {target}=={target_class}')
    plt.ylim(top=ylim)
    plt.show()
    return


def barplots_low_card_feat_by_target_eq_class(
        target: str,
        target_class,
        dataframe,
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
    :param dataframe: pandas dataframe target column
    :param nunique: limit features to those which have <= this value unique
        values, optional, default: 15
    :param ylim: upper limit of y-axis, optional, default: 0.7
    :param figsize: size of matplotlib figure, optional, default: (9, 6)

    :returns: nothing
    """
    assert target in dataframe.columns, \
        f'TARGET: {target} NOT FOUND IN DATAFRAME.columns'
    for feature in dataframe.columns[
            (dataframe.nunique() <= nunique) & (dataframe.nunique() > 1)] \
            .drop([target], errors='ignore'):
        barplot_feat_by_target_eq_class(
            feature,
            target,
            target_class,
            dataframe,
            ylim=ylim,
            figsize=figsize
        )
    return


def value_counts(dataframe, features):
    """
    Display value counts and normalized value counts together for one or more
        features in a pandas dataframe.

    :param dataframe: pandas dataframe
    :param features: a feature or list of features found in the dataframe

    :returns: nothing
    """
    if not isinstance(features, (list, set, tuple)):
        features = [features]
    for feature in features:
        assert feature in dataframe.columns, \
            f'{feature} not found in dataframe'
        df = pd.DataFrame({
            'count': dataframe[feature].value_counts().sort_index(),
            'percentage': dataframe[feature].value_counts(normalize=True)
            .sort_index()
        })
        df.index.name = feature
        display(df.sort_values(by='count', ascending=False))
    return
