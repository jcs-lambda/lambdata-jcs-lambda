from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lambdata_jcslambda",
    version="0.1.1",
    author="jcs-lambda",
    author_email="57103874+jcs-lambda@users.noreply.github.com",
    description="Example package for lambda school DS Unit 3",
    long_description=long_description,
    long_description_content_type="text/markdown", # required if using a md file for long desc
    license="MIT",
    url="https://github.com/jcs-lambda/lambdata-jcs-lambda",
    keywords="lambda school",
    packages=find_packages(), # ["my_lambdata"]
    python_requires = ">=3.7.5"
)