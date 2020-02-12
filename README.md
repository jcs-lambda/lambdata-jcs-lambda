# lambdata-jcs-lambda
 python package of utility functions

 current version: 0.3.9

## Installation

    pip install -i https://test.pypi.org/simple/ lambdata-jcslambda

## Use

    from lambdata_jcslambda import df_utils
    from lambdata_jcslambda import Explorator
    train, val, test = df_utils.tvt_split(df)
    xdf = Explorator(df)
    xdf.value_counts(['column1', 'column2'])

## Example Usage
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcs-lambda/lambdata-jcs-lambda/blob/master/notebooks/lambdata_testbed.ipynb 'Test Notebook on Colab')
