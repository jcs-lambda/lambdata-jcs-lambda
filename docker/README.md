# lambdata-jcs-lambda docker image

The Dockerfile in this folder can be used to create a docker image that
is ready to use the lambdata_jcslambda package.

## What is docker

[Docker](https://docker.com/) is a technology that uses containers for
building and sharing apps easily.

It allows a developer to create a single distributable unit that contains
everything needed for an app to run correctly. A developer would use
containers to isolate and encapsulate an entire project into one entity.
This allows for simplified sharing, collaborating, and use throughout all
stages of a product's life-cycle. Containers can be used to provide a known
good base-line starting point for contributors to or new users of a project.

# Using this Dockerfile

### Build the image

    docker build . -t python_lambdata_jcslambda

### Start a container

    docker run -it python_lambdata_jcslambda /bin/bash

### Launch python interpreter

    python3

### Import and use this library

    import lambdata_jcslambda.df_utils
    dates = my.pd.date_range(start='2017-01-01', end='2017-12-31', freq='W-WED')
    dates = my.pd.DataFrame(dates, columns=['Date'])
    dates.head()
    df_utils.extract_date_parts(dates, 'Date', simple=False).head()
