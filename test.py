#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing the performance of the model
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

from CONFIG import *
import joblib
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly
from models import HierarchicalClassifier

from colorama import init
from termcolor import *
init()

model = HierarchicalClassifier.from_finetuned(checkpoint)

title = 'amazing product'
text = 'my hairs have gained volume after 2 weeks into using this product'
cprint(model.out_of_sample_prediction(title, text), 'blue')
# >> [array(['beauty'], dtype=object), 
# array(['hair care'], dtype=object), 
# array(['massage relaxation'], dtype=object)]

title = 'do not buy this'
text = 'got broken in 2 weeks'
cprint(model.out_of_sample_prediction(title, text), 'blue')
# >> [array(['baby products'], dtype=object), 
# array(['games'], dtype=object), 
# array(['rare'], dtype=object)]

########################

true1 = joblib.load('./classifier/data/cat1_true.joblib')
pred1 = joblib.load('./classifier/data/cat1_pred.joblib')
M1 = confusion_matrix(true1, pred1)

true2 = joblib.load('./classifier/data/cat2_true.joblib')
pred2 = joblib.load('./classifier/data/cat2_pred.joblib')
M2 = confusion_matrix(true2, pred2)

true3 = joblib.load('./classifier/data/cat3_true.joblib')
pred3 = joblib.load('./classifier/data/cat3_pred.joblib')
M3 = confusion_matrix(true3, pred3)



label_encoders1 = joblib.load('./classifier/data/label_encoder_cat1.joblib')
label_encoders2 = joblib.load('./classifier/data/label_encoder_cat2.joblib')
label_encoders3 = joblib.load('./classifier/data/label_encoder_cat3.joblib')



fig = px.imshow(M1,
                x=label_encoders1.classes_,
                y=label_encoders1.classes_,
                color_continuous_scale='turbo')
plotly.offline.plot(fig, filename='./classifier/report/cm1.html')


fig = px.imshow(M2,
                x=label_encoders2.classes_,
                y=label_encoders2.classes_,
                color_continuous_scale='turbo')
plotly.offline.plot(fig, filename='./classifier/report/cm2.html')


fig = px.imshow(M3,
                x=label_encoders3.classes_,
                y=label_encoders3.classes_,
                color_continuous_scale='turbo')
plotly.offline.plot(fig, filename='./classifier/report/cm3.html')