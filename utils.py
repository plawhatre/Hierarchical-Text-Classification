#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions required in the task
"""

from __future__ import print_function

__version__ = "1.0"
__author__ = "Prashant Lawhatre"
__license__ = "GPL"
__email__ = "prashantlawhatre@gmail.com"

import pandas as pd
import numpy as np

import nltk
from nlpaug.augmenter.word import AntonymAug, SynonymAug, RandomWordAug
import nlpaug.model.word_stats as nmw
import os
import joblib


nltk.download('averaged_perceptron_tagger', quiet=True);
nltk.download('wordnet', quiet=True);
nltk.download('omw-1.4', quiet=True);


def augment(text, times):
	l1 = AntonymAug().augment(text, times)
	l2 = SynonymAug().augment(text, times)
	l3 = RandomWordAug().augment(text, times)

	if not isinstance(l1, list):
		l1 = [l1]

	if not isinstance(l2, list):
		l2 = [l2]

	if not isinstance(l3, list):
		l3 = [l3]

	l = [text] + l1 + l2 + l3 
	
	return l

def loss2DF(losses):
	epochs, batches = len(losses), len(losses[0])
	df = pd.DataFrame({}, columns=['Epoch', 'Batch', 'Loss'])

	for epoch in range(epochs):
		for batch in range(batches):
			df = df.append(
				{'Epoch':int(epoch), 'Batch':int(batch), 'Loss':losses[epoch][batch].numpy()},
				ignore_index=True)

	return df

def save_results(true, pred, msg):
	p = './classifier/data/' + msg + '_'
	
	joblib.dump(true, p+'true.joblib')
	joblib.dump(pred, p+'pred.joblib')