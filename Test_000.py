# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:01:32 2019

@author: Ian 
"""

#TODO
# fix oob samples - split with random shuffle, then resample on training half?
#  -- switch back to training on full dataset after oob fix
# fix hold out test Pr@Re50
# add single tree, and combine in permutation test graph
# not sure about permutation test accuracy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.tree import DecisionTreeClassifier
from numpy import ravel
import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

start_time = time.time()

# load dataset
#Indata = pd.read_csv("csv_result-Descriptors_Training.csv")
Indata = pd.read_csv('csv_result-Descriptors_Calibration.csv')
data = Indata.values
raw_data_df = data[:,1:-1];
raw_class_df = data[:,-1];

Intdata = pd.read_csv('csv_result-Descriptors_Calibration.csv')
tdata = Intdata.values
test_data_df = tdata[:,1:-1];
test_class_df = tdata[:,-1];

# #############################################################################
#   STEP 1: run model on training data to get apparent errors
# #############################################################################
print('\n---- Apparent Error - Train on Training Data, Test on Test Data ----')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
data_train, data_verif, class_train, class_verif = train_test_split(raw_data_df, raw_class_df, test_size = 0.3, random_state = 2, stratify = raw_class_df)

#data_verif, class_train, class_verif
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
clf.fit(data_train, ravel(class_train))
prediction = clf.predict(data_verif)
pred = clf.predict_proba(data_verif)

# keep probabilities for the positive outcome only
pred = pred[:, 1]

precision, recall, thresholds = sklm.precision_recall_curve(class_verif, pred, pos_label="P")
average_precision = sklm.average_precision_score(class_verif, pred, pos_label="P")

Re50 = [0.5]
PrecisionAtRe50 = np.interp(Re50, recall, precision, period=0.01)

plt.figure()
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))
plt.plot(Re50, PrecisionAtRe50, '-x')
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
# **** END INSERT MODEL HERE ************************************************************
# **** END INSERT MODEL HERE ************************************************************
precision_app = PrecisionAtRe50
print('Apparent Precision (Pr@Re50): ', *precision_app)

# #############################################################################
#   STEP 2: bootstrap  model on training data to get apparent errors
# #############################################################################
print('\n---- Estimate Error - Bootstrapping on Test Data -------------------')

# configure bootstrap
n_iterations = 3 #1000
n_size = int(len(data) * 0.5) #0.5

y_scores = np.array([0.5])

# run bootstrap
stats_precision = list()
for i in range(n_iterations):
	# prepare train and test sets
#    train = resample(data, n_samples=n_size, stratify=data[:,-1])
    train = resample(data, n_samples=n_size)
    test = np.array([x for x in data if x.tolist() not in train.tolist()])

    data_train = train[:,1:-1];
    class_train = train[:,-1];
    data_test = test[:,1:-1];
    class_test = test[:,-1];
    
	# evaluate model
    clfb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
    clfb.fit(data_train, ravel(class_train))
    prediction = clfb.predict(data_verif)
    pred = clfb.predict_proba(data_verif)
    
    # keep probabilities for the positive outcome only
    pred = pred[:, 1]
    
    precision, recall, thresholds = sklm.precision_recall_curve(class_verif, pred, pos_label="P")
    
    Re50 = [0.5]
    PrecisionAtRe50 = np.interp(Re50, recall, precision, period=0.01)
    print('Pr@Re50 = ', *PrecisionAtRe50)

# #############################################################################
# Bootstrapping Results 
print('\n----  Bootstrapping Results ----------------------------------------')
print('Precision Mean: ',np.mean(stats_precision), ', Standard Deviation: ', np.std(stats_precision))
# confidence intervals
#alpha = 0.95
#p = ((1.0-alpha)/2.0) * 100
#lower = max(0.0, np.percentile(stats_precision, p))
#p = (alpha+((1.0-alpha)/2.0)) * 100
#upper = min(1.0, np.percentile(stats_precision, p))
#print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

# plot scores
plt.figure()
plt.hist(stats_precision, label='Permutation scores', edgecolor='black')
plt.show()

# #############################################################################
#   STEP 3: Calculate 0.632 Bootstrapping value
# #############################################################################
print('\n---- 0.632 Bootstrapping value -------------------------------------')
bootstrap632 = 0.632*np.mean(stats_precision) + 0.368*precision_app
print('0.632 Bootstrapping Estimate = ', *bootstrap632)

# #############################################################################
#   STEP 4: Run Holdout Test on Test data
# #############################################################################
print('\n---- Holdout Test on Test Data -------------------------------------')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
prediction = clf.predict(test_data_df)
pred = clf.predict_proba(test_data_df)

# keep probabilities for the positive outcome only
pred = pred[:, 1]

precision, recall, thresholds = sklm.precision_recall_curve(test_class_df, pred, pos_label="P")
average_precision = sklm.average_precision_score(test_class_df, pred, pos_label="P")

#Re50 = [0.5]
#PrecisionAtRe50 = np.interp(Re50, recall, precision, period=0.01)
idx = (np.abs(recall-0.5)).argmin()
PrecisionAtRe50 = [precision[idx]]

plt.figure()
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))
plt.plot(Re50, PrecisionAtRe50, '-x')
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
# **** END INSERT MODEL HERE ************************************************************
# **** END INSERT MODEL HERE ************************************************************
precision_test = PrecisionAtRe50
print('Test Precision (Pr@Re50): ', *precision_test)

# #############################################################################
#   STEP 5: Is classifier better than random? Permutation Test 
# #############################################################################
print('\n---- Permutation on Test Data --------------------------------------')
#https://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py
cv = StratifiedKFold(2)
score, permutation_scores, pvalue = permutation_test_score(clf, test_data_df, test_class_df, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

# #############################################################################
# View histogram of permutation scores
plt.figure()
plt.hist(permutation_scores, 20, label='Permutation scores', edgecolor='black')
ylim = plt.ylim()

plt.plot(2 * [score], ylim, '--g', linewidth=3, label='Classification Score' ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()


print("Execution took %s seconds" % (time.time() - start_time))