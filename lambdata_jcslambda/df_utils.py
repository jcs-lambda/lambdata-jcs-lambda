"""Utility functions for working with dataframes

Functions:

tvt_split: split a dataframe into 3 sets

expand_date_parts: replace dates in a dataframe column with date parts

describe: create a summary of a dataframe

barplot_feat_by_target_eq_class: barplot of feature1 by feature2==X

barplots_low_card_feat_by_target_eq_class: barplot for each low-cardinality
feature by some_feature==X

value_counts: actual and normalized value counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display


def tvt_split(df, target: str='', random_state=None):
    """Split a dataframe into train, validation, and test sets.

    Parameters:

    df: pandas dataframe

    target: name of a column passed as stratify parameter to
    sklearn.model_selection.train_test_split(), optional

    random_state: int or None, optional, default: None

    Returns:

    tuple of 3 dataframes: (train, validation, test)
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


def expand_date_parts(dataframe, date_column, simple=True):
    """Replace single date-like column with columns representing date parts.

    Parameters:

    dataframe: pandas dataframe

    date_column: name of date-like column to send to pandas.to_datetime()

    simple: True - convert to year, month, and day. False - also include
    day_of_week, day_of_year, week, and quarter, optional, default: False

    Returns:

    pandas dataframe with date_column removed and other columns added
    """
    if date_column not in dataframe.columns:
        raise KeyError(
            f'dataframe does not contain column: {date_column}'
        )
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
    """Return a dataframe with summary description.

    Parameters:

    dataframe: pandas dataframe

    formatter: numpy formatter used in numpy.array2string. optional,
    pass None to use default numpy formatting.

    Returns

    pandas dataframe
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
        figsize=(9, 6)):
    """Display barplot of x feature by (y target == target_class).

    Parameters:

    feature: name of column to use as x feature

    target: name of column to use as y target

    target_class: value tested for equality with each value in y target

    dataframe: pandas dataframe

    ylim: upper limit of y-axis, optional, default: 0.7

    figsize: size of matplotlib figure, optional, default: (9, 6)

    Returns:

    nothing
    """
    if feature not in dataframe.columns:
        raise KeyError(
            f'dataframe does not contain feature column: {feature}'
        )
    if target not in dataframe.columns:
        raise KeyError(
            f'dataframe does not contain target column: {target}'
        )
    if target_class not in dataframe[target].unique():
        raise ValueError(
            f"dataframe['{target}'] does not contain" +
            f" target_class: {target_class}"
        )
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
        figsize=(9, 6)):
    """Display barplots of x feature by (y target == target_class)
    for features in a dataframe with <= nunique unique values.

    Parameters:

    target: name of column to use as y target

    target_class: value whose equality is tested with each value in target

    dataframe: pandas dataframe

    nunique: max number of unique values for features to plot

    ylim: upper limit of y-axis, optional, default: 0.7

    figsize: size of matplotlib figure, optional, default: (9, 6)

    Returns:

    nothing
    """
    if target not in dataframe.columns:
        raise KeyError(
            f'dataframe does not contain target column: {target}'
        )
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
    """Display actual and normalized value counts for one or more features.

    Displayed with ipython for multiple outputs in one notebook cell.

    Parameters:

    dataframe: pandas dataframe

    features: feature name or list-like of feature names

    Returns:

    nothing
    """
    if not isinstance(features, (list, set, tuple, np.ndarray, pd.Series)):
        features = [features]
    for feature in features:
        if feature not in dataframe.columns:
            print(f'dataframe does not contain feature: {feature}')
            continue
        df = pd.DataFrame({
            'count': dataframe[feature].value_counts().sort_index(),
            'percentage': dataframe[feature].value_counts(normalize=True)
            .sort_index()
        })
        df.index.name = feature
        display(df.sort_values(by='count', ascending=False))
    return
