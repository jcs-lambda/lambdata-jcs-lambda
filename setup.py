"""Create lambdata_jcslambda package."""

from setuptools import find_packages, setup

REQUIRES = [
    'numpy',
    'pandas',
    'ipython',
    'matplotlib',
    'seaborn',
    'ipywidgets',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lambdata_jcslambda",
    version="0.4.2",
    author="jcs-lambda",
    author_email="57103874+jcs-lambda@users.noreply.github.com",
    description="Example package for lambda school DS Unit 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/jcs-lambda/lambdata-jcs-lambda",
    keywords="lambda school",
    packages=find_packages(exclude=['tests']),
    python_requires = ">=3.5",
    install_requires = REQUIRES,
)