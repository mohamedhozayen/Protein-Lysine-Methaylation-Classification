#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:07:47 2019

@author: mohamedhozayen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

Figures = 'Figures/'
Files = 'Files/'

val = pd.read_csv('csv_result-Descriptors_Calibration.csv', sep=',')
train = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 

# =============================================================================
# Class probability
# =============================================================================
from __future__ import division
total_p = sum(train['class'].str.count('P')) + sum(val['class'].str.count('P'))
total_n = sum(train['class'].str.count('N')) + sum(val['class'].str.count('N'))
p_prob = total_p / (total_n + total_p)


# =============================================================================
# correlation
# =============================================================================
train.corr(method='kendall').to_csv(Files + 'Raw-Matrix-Training-kendall.csv')
train.corr(method='spearman').to_csv(Files + 'Raw-Matrix-Training-spearman.csv')
train.corr(method='pearson').to_csv(Files + 'Raw-Matrix-Training-pearson.csv')

f = plt.figure(figsize=(15, 10))
plt.matshow(train.corr(method='kendall'), fignum=f.number, cmap=plt.cm.bwr)
plt.xticks(range(train.shape[1]), train.columns, fontsize=10, rotation=80)
plt.yticks(range(train.shape[1]), train.columns, fontsize=10)
cb = plt.colorbar()
#cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix - Training - kendall', fontsize=12, y=1.2);
plt.savefig(Figures + 'Correlation-Matrix-Training-kendall.pdf')


# =============================================================================
# Heat map
# =============================================================================
from sklearn import preprocessing
data = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',')
data['class'] = preprocessing.LabelBinarizer().fit(data['class']).transform(data['class'])

plt.figure(figsize=(25,25))
cor = data.corr()
import seaborn as sns
sns_plot = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
sns_plot.get_figure().savefig(Figures + 'heatmap-pearsons'  + '.pdf')





























