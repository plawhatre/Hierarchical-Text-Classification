#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pre-process the dataset

"""

from __future__ import print_function

__version__ = "1.0"
__author__ = "Prashant Lawhatre"
__license__ = "GPL"
__email__ = "prashantlawhatre@gmail.com"

import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from glob import glob
import pandas as pd

from wordcloud import WordCloud, STOPWORDS
from stop_words import get_stop_words
import re
import string
import nltk
import joblib
from models import Extractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import augment

import CONFIG

N_AUGS = CONFIG.n_augs


nltk.download('averaged_perceptron_tagger', quiet=True);
nltk.download('wordnet', quiet=True);

if not os.path.exists('./classifier'):
	os.makedirs('./classifier/data')
	os.makedirs('./classifier/model')
	os.makedirs('./classifier/report')

nltk.download('wordnet', quiet=True);


# Read data
file = glob('./data/*.csv')[0]
df = pd.read_csv(file)

# Drop un-informative columns
df.drop(['productId', 'userId', 'Time'], axis=1, inplace=True)

# Group instances on Cat3  
df_group = df.groupby(df.Cat3).count().Cat1




# Cat3
rare_instance_classes3 = df_group.loc[df.groupby(df.Cat3).count().Text<=10].index 
rare3 = df.Cat3.apply(lambda x: x in rare_instance_classes3)

df['Cat3'].loc[rare3] = 'rare'

# Group instances on Cat3
df_group = df.groupby(df.Cat2).count().Cat1
# Cat2

rare_instance_classes2 = df_group.loc[df.groupby(df.Cat2).count().Text<=20].index 
rare2 = df.Cat2.apply(lambda x: x in rare_instance_classes2)

df['Cat2'].loc[rare2] = 'rare'

joblib.dump(rare3, './classifier/data/rare3.joblib')
joblib.dump(rare2, './classifier/data/rare2.joblib')

joblib.dump(rare_instance_classes3, './classifier/data/rare_instance_classes3.joblib')
joblib.dump(rare_instance_classes2, './classifier/data/rare_instance_classes2.joblib')




# PRE-PROCESSING
stopwords = set(get_stop_words('english')) 

# remove numbers
df.Title = df.Title.apply(
	lambda x: re.sub('\d+', '', str(x)))
df.Text = df.Text.apply(
	lambda x: re.sub('\d+', '', str(x)))

# convert to lower case
df.Title = df.Title.apply(
	lambda x: str(x).lower())
df.Text = df.Text.apply(
	lambda x: str(x).lower())

# remove stopwords
df.Title = df.Title.apply(
	lambda x: " ".join([word for word in x.split(" ") if word not in stopwords]))
df.Text = df.Text.apply(
	lambda x: " ".join([word for word in x.split(" ") if word not in stopwords]))

# remove punctuations
df.Title = df.Title.apply(
	lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df.Text = df.Text.apply(
	lambda x: x.translate(str.maketrans('', '', string.punctuation)))



# Label Encoding
label_encoder_cat1 = LabelEncoder().fit(df.Cat1)
label_encoder_cat2 = LabelEncoder().fit(df.Cat2)
label_encoder_cat3 = LabelEncoder().fit(df.Cat3)

df.Cat1 = label_encoder_cat1.transform(df.Cat1)
df.Cat2 = label_encoder_cat2.transform(df.Cat2)
df.Cat3 = label_encoder_cat3.transform(df.Cat3)

joblib.dump(label_encoder_cat1, './classifier/data/label_encoder_cat1.joblib')
joblib.dump(label_encoder_cat2, './classifier/data/label_encoder_cat2.joblib')
joblib.dump(label_encoder_cat3, './classifier/data/label_encoder_cat3.joblib')




# Train test split
df_train, df_test = train_test_split(
    df, test_size=0.25, random_state=42, stratify=df.Cat3)



# Data Augmentation for Cat3
df_train.Text.loc[rare3] = df_train.Text.loc[rare3].apply(
	lambda x: augment(x, N_AUGS))

df_train = df_train.explode('Text').reset_index(drop=True)

joblib.dump(df_train, './classifier/data/raw_df_train.csv')
joblib.dump(df_test, './classifier/data/raw_df_test.csv')



# Extractor
checkpoint = 'distilbert-base-uncased'
extract = Extractor(checkpoint)

df_train.Title = df_train.Title.apply(lambda x: extract(x))
df_train.Text = df_train.Text.apply(lambda x: extract(x))

df_test.Title = df_test.Title.apply(lambda x: extract(x))
df_test.Text = df_test.Text.apply(lambda x: extract(x))

joblib.dump(df_train, './classifier/data/df_train.csv')
joblib.dump(df_test, './classifier/data/df_test.csv')