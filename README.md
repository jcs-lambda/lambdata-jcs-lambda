# lambdata-jcs-lambda
 python package of utility functions

 current source version: 0.4.3
 
 current test.pypi version: 0.4.3

## Installation

    pip install -i https://test.pypi.org/simple/ lambdata-jcslambda

## Use

    from lambdata_jcslambda import df_utils
    from lambdata_jcslambda import Explorator
    train, val, test = df_utils.tvt_split(df)
    xdf = Explorator(df)
    xdf.value_counts(['column1', 'column2'])

## Test it out yourself
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcs-lambda/lambdata-jcs-lambda/blob/master/notebooks/lambdata_testbed.ipynb 'Test Notebook on Colab')
