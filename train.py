#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training the model on the task

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
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers import TFAutoModel
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from models import HierarchicalClassifier
from sklearn.metrics import classification_report
import plotly
import plotly.express as px

from utils import loss2DF, save_results

from colorama import init
from termcolor import *
init()

import CONFIG

CHECKPOINT = CONFIG.checkpoint

# Phase 1
LEARNING_RATE_1 = CONFIG.learning_rate_1
BETA_1_1 = CONFIG.beta_1_1
BETA_2_1 = CONFIG.beta_2_1
EPSILON_1 = CONFIG.epsilon_1

EPOCHS_1 = CONFIG.epochs_1
BATCH_SIZE_1 = CONFIG.batch_size_1

# Phase 2
LEARNING_RATE_2 = CONFIG.learning_rate_2
BETA_1_2 = CONFIG.beta_1_2
BETA_2_2 = CONFIG.beta_2_2
EPSILON_2 = CONFIG.epsilon_2

EPOCHS_2 = CONFIG.epochs_2
BATCH_SIZE_2 = CONFIG.batch_size_2

# Phase 3
LEARNING_RATE_3 = CONFIG.learning_rate_3
BETA_1_3 = CONFIG.beta_1_3
BETA_2_3 = CONFIG.beta_2_3
EPSILON_3 = CONFIG.epsilon_3

EPOCHS_3= CONFIG.epochs_3
BATCH_SIZE_3 = CONFIG.batch_size_3



if __name__ == '__main__':

	# Read data
	df_train = joblib.load('./classifier/data/df_train.csv')
	df_test = joblib.load('./classifier/data/df_test.csv')

	label_encoders = []

	label_encoders.append(joblib.load('./classifier/data/label_encoder_cat1.joblib'))
	label_encoders.append(joblib.load('./classifier/data/label_encoder_cat2.joblib'))
	label_encoders.append(joblib.load('./classifier/data/label_encoder_cat3.joblib'))
	


	Model = HierarchicalClassifier(CHECKPOINT, label_encoders)


	# Phase 1: Cat1 
	optimizer1 = tf.keras.optimizers.Adam(LEARNING_RATE_1,
		BETA_1_1, 
		BETA_2_1,
		EPSILON_1)

	Loss = Model.train(df_train, optimizer1, EPOCHS_1, BATCH_SIZE_1, '1')
	preds = Model.predict(df_test)
	print(report:=classification_report(df_test.Cat1, preds[0][0]))

	with open('./classifier/report/model1.txt', 'w') as f:
		f.write(str(report))

	df_loss = loss2DF(Loss)
	fig = px.line(df_loss, x=df_loss.index.values, y="Loss", color='Epoch')
	plotly.offline.plot(fig, filename='./classifier/report/loss1.html')

	save_results(df_test.Cat1, preds[0][0], 'cat1')


	# Phase 2: Cat2 
	optimizer2 = tf.keras.optimizers.Adam(LEARNING_RATE_2,
		BETA_1_2, 
		BETA_2_2,
		EPSILON_2)

	Loss = Model.train(df_train, optimizer2, EPOCHS_2, BATCH_SIZE_2, '2')
	preds = Model.predict(df_test)
	print(report:=classification_report(df_test.Cat2, preds[0][1]))

	with open('./classifier/report/model2.txt', 'w') as f:
		f.write(str(report))

	df_loss = loss2DF(Loss)
	fig = px.line(df_loss, x=df_loss.index.values, y="Loss", color='Epoch')
	plotly.offline.plot(fig, filename='./classifier/report/loss2.html')

	save_results(df_test.Cat2, preds[0][1], 'cat2')


	# Phase 3: Cat3
	optimizer3 = tf.keras.optimizers.Adam(LEARNING_RATE_3,
		BETA_1_3, 
		BETA_2_3,
		EPSILON_3)

	Loss = Model.train(df_train, optimizer3, EPOCHS_3, BATCH_SIZE_3, '3')
	preds = Model.predict(df_test)
	print(report:=classification_report(df_test.Cat3, preds[0][2]))

	with open('./classifier/report/model3.txt', 'w') as f:
		f.write(str(report))

	df_loss = loss2DF(Loss)
	fig = px.line(df_loss, x=df_loss.index.values, y="Loss", color='Epoch')
	plotly.offline.plot(fig, filename='./classifier/report/loss3.html')

	save_results(df_test.Cat3, preds[0][2], 'cat3')