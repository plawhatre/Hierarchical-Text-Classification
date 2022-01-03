#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis

"""

from __future__ import print_function

__version__ = "1.0"
__author__ = "Prashant Lawhatre"
__license__ = "GPL"
__email__ = "prashantlawhatre@gmail.com"

from glob import glob
import pandas as pd
import plotly.express as px

# Read data
file = glob('./data/*.csv')[0]
df = pd.read_csv(file)

# Number of data instances
print("Numer of Data instances", df.shape[0], '\n')
# >> 10000

# Number of features
print("Numer of features", df.shape[1], '\n')
# >> 8

# Columns in dataframe
print("Columns: ", df.columns, '\n')
# >> ['productId', 'Title', 'userId', 'Time', 'Text', 'Cat1', 'Cat2', 'Cat3'],

# Unique product ids
print("Number of unique product ids", df.productId.nunique(), '\n')
# >> 6865

# Unique user ids
print("Number of user product ids", df.userId.nunique(), '\n')
# >> 9716

# Cat1
print("Levels in Cat1", df.Cat1.nunique(), '\n')
# >> 6

# Cat2 
print("Levels in Cat2", df.Cat2.nunique(), '\n')
# >> 64

# Cat3
print("Levels in Cat3", df.Cat3.nunique(), '\n')
# >> 377

# Datatypes
print("Dataset dtypes", df.dtypes)


fig = px.histogram(df, x="Cat1")
fig.show()

fig = px.histogram(df, x="Cat2")
fig.show()

fig = px.histogram(df, x="Cat3")
fig.show()