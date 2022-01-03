#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model & Extractor class for the task

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
import tensorflow as tf

import joblib
from colorama import init
from termcolor import *
init()

class Extractor:
	def __init__(self, checkpoint):
		if not os.path.exists('./pretrained'):
			os.makedirs('./pretrained/tokenizer')
			os.makedirs('./pretrained/extractor')

		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, 
			cache_dir='./pretrained/tokenizer',
			local_files_only=True)

		self.model = TFAutoModel.from_pretrained(checkpoint,
			output_hidden_states=True,
			cache_dir='./pretrained/extractor',
			local_files_only=True)

		cprint('Extractor Created !!', 'green')

	def __call__(self, X):

		X = self.tokenizer(X, 
			padding=True, 
			truncation=True,
			return_tensors='tf')

		y = self.model(X).last_hidden_state
		y = tf.math.reduce_mean(y, axis=1)
		return y

class HierarchicalClassifier(tf.keras.layers.Layer):
	def __init__(self, checkpoint, label_encoders, from_finetuned=False, **kwargs):
		super(HierarchicalClassifier, self).__init__()

		self.extractor = Extractor(checkpoint)
		self.label_encoders = label_encoders
		
		if from_finetuned == False:
			self.model1 = tf.keras.layers.Dense(label_encoders[0].classes_.shape[0])
			self.model2 = tf.keras.layers.Dense(label_encoders[1].classes_.shape[0])
			self.model3 = tf.keras.layers.Dense(label_encoders[2].classes_.shape[0])

		else:
			self.model1 = kwargs['model1']
			self.model2 = kwargs['model2']
			self.model3 = kwargs['model3']

		cprint('Model Created !!', 'green')

	@classmethod
	def from_finetuned(cls, checkpoint):
		label_encoders = []
		dict_models = {}

		label_encoders.append(
			joblib.load('./classifier/data/label_encoder_cat1.joblib'))
		label_encoders.append(
			joblib.load('./classifier/data/label_encoder_cat2.joblib'))
		label_encoders.append(
			joblib.load('./classifier/data/label_encoder_cat3.joblib'))

		dict_models.update(
			{'model1':joblib.load('./classifier/model/model1.joblib')})
		
		dict_models.update(
			{'model2':joblib.load('./classifier/model/model2.joblib')})
		
		dict_models.update(
			{'model3':joblib.load('./classifier/model/model3.joblib')})
		

		return cls(checkpoint, label_encoders, from_finetuned=True, **dict_models)


	def call(self, title, text, phase):

		if int(phase) >= 1:
			X1 = tf.concat([title, text], axis=-1)
			logits1 = self.model1(X1)
			Y1 = tf.keras.activations.softmax(logits1)
			
			if int(phase) >= 2:
				X2 = tf.concat([X1, Y1], axis=-1)
				logits2 = self.model2(X2)
				Y2 = tf.keras.activations.softmax(logits2)

				if int(phase) >= 3:
					X3 = tf.concat([X2, Y2], axis=-1)
					logits3 = self.model3(X3)
					Y3 = tf.keras.activations.softmax(logits3)

					return Y3
				return Y2
			return Y1

		else:
			X1 = tf.concat([title, text], axis=-1)
			logits1 = self.model1(X1)
			Y1 = tf.keras.activations.softmax(logits1)

			X2 = tf.concat([X1, Y1], axis=-1)
			logits2 = self.model2(X2)
			Y2 = tf.keras.activations.softmax(logits2)

			X3 = tf.concat([X2, Y2], axis=-1)
			logits3 = self.model3(X3)
			Y3 = tf.keras.activations.softmax(logits3)

			return Y1, Y2, Y3

	def predict(self, df):
		cprint('Making predictions......', 'green')

		title = list(df.Title.values)
		text = list(df.Text.values)

		out = self(title, text, '0')

		output = []
		labels = []

		for i, encoder in enumerate(self.label_encoders):
			output.append(tf.math.argmax(out[i], axis=-1).numpy())
			labels.append(encoder.inverse_transform(output[-1]))

		return output, labels

	def out_of_sample_prediction(self, title, text):
		cprint('Making predictions......', 'green')

		title = [self.extractor(title)]
		text = [self.extractor(text)]

		out = self(title, text, '0')

		output = []
		labels = []

		for i, encoder in enumerate(self.label_encoders):
			output.append(tf.math.argmax(out[i], axis=-1).numpy())
			labels.append(encoder.inverse_transform(output[-1]))

		return labels


	def loss(self, Y_true, Y_pred):
		scce = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=False)
		return scce(Y_true, Y_pred)

	def backprop(self, X, Y, optimizer, phase):
		with tf.GradientTape() as tape:
			y_pred = self(*X, phase)
			loss = self.loss(Y, y_pred)

		gradients = tape.gradient(loss, self.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return loss

	def train(self, df, optimizer, epochs=1, batch_size=32, phase='1'):

		title = list(df.Title.values)
		text = list(df.Text.values)

		Y1 = df.Cat1.values

		if int(phase) > 1:
			
			self.model1 = joblib.load('./classifier/model/model1.joblib')

			Y2 = df.Cat2.values

			if int(phase) > 2:
				
				self.model2 = joblib.load('./classifier/model/model2.joblib')

				Y3 = df.Cat3.values
				
				if int(phase) > 3:
					
					return None

		n_batches = df.shape[0] // batch_size + 1

		Loss = []
		for i in range(epochs):
			loss = []
			for j in range(n_batches):
				X_title = title[(j*batch_size):((j+1)*batch_size)]
				X_text = text[(j*batch_size):((j+1)*batch_size)]

				X = [X_title, X_text]

				if phase == str(1):
					self.model1.trainable = True
					self.model2.trainable = False
					self.model3.trainable = False

					y = Y1[(j*batch_size):((j+1)*batch_size)]

				elif phase == str(2):
					self.model1.trainable = False
					self.model2.trainable = True
					self.model3.trainable = False

					y = Y2[(j*batch_size):((j+1)*batch_size)]

				elif phase == str(3):
					self.model1.trainable = False
					self.model2.trainable = False
					self.model3.trainable = True

					y = Y3[(j*batch_size):((j+1)*batch_size)]

				else:
					return 
				
				l = self.backprop(X, y, optimizer, phase)
				loss.append(l)

				cprint(f"Phase: {phase}, Epochs: {i+1}/{epochs}, Batch: {j}/{n_batches+1}, Loss: {l}", 
					'magenta')

			Loss.append(loss)

		if int(phase) == 1:
			joblib.dump(self.model1,'./classifier/model/model1.joblib')
		
		if int(phase) == 2:
			joblib.dump(self.model2,'./classifier/model/model2.joblib')
		
		if int(phase) == 3:
			joblib.dump(self.model3,'./classifier/model/model3.joblib')

		return Loss