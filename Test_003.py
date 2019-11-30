# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:01:32 2019

@author: Ian 
"""

#TODO
# add single tree, and combine in permutation test graph

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

import preprocessing as prc

start_time = time.time()


# load datasets
inTrainData = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
inTrainData = inTrainData.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
inTrainData = prc.detect_outlier_iterative_IQR(inTrainData).dropna()
# inTrainData = prc.detect_outlier_iterative_IQR(inTrainData).fillna(0)
# Split into data and class
train_data = inTrainData.drop(['class'], axis=1)
train_class = inTrainData['class']

inTestData = pd.read_csv('csv_result-Descriptors_Calibration.csv', sep=',') 
inTestData = inTestData.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
inTestData = prc.detect_outlier_iterative_IQR(inTestData).dropna()
# inTestData = prc.detect_outlier_iterative_IQR(inTestData).fillna(0)
# Split into data and class
test_data = inTestData.drop(['class'], axis=1)
test_class = inTestData['class']


# configure bootstrap
n_iterations = 20;
n_size = int(len(train_data) * 0.5) #0.5


# ================================================================================================================================================
# ================================================================================================================================================
# DECISION TREE SECTION
# ================================================================================================================================================
# ================================================================================================================================================
print('\n=== DECISION TREE ==================================================')
# #############################################################################
#   STEP 1: run model on training data to get apparent errors
# #############################################################################
print('\n---- Apparent Error - Train on Training Data, Test on Test Data ----')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
# NOTE Stratified KFold!
kf = StratifiedKFold(n_splits=5, shuffle = True)
kf.get_n_splits(train_data)


class_true = []
class_pred = []
class_prob = []
for train_index, test_index in kf.split(train_data, train_class):
	# Train and test model  
	dt = DecisionTreeClassifier(max_depth = 8, class_weight = {1: 5, 0: 1}, max_leaf_nodes=100)
	this_train_data, this_test_data= train_data.iloc[train_index], train_data.iloc[test_index]
	this_train_class, this_test_class = train_class.iloc[train_index], train_class.iloc[test_index]
	dt.fit(this_train_data, this_train_class)

	class_true.extend(this_test_class)
	class_pred.extend(dt.predict(this_test_data).tolist())
	class_prob.extend(dt.predict_proba(this_test_data)[::,1])

precision, recall, thresholds = sklm.precision_recall_curve(class_true, class_prob)
average_precision = sklm.average_precision_score(class_true, class_prob)
# **** END INSERT MODEL HERE ************************************************************
# **** END INSERT MODEL HERE ************************************************************
    
    
PrecisionAtRe50_DT = np.max(precision[recall>=0.5])

plt.figure()
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))
plt.plot([0, 1], [PrecisionAtRe50_DT, PrecisionAtRe50_DT], '-r', linewidth=1, label="Pr@Re50 = {0:.4f}".format(PrecisionAtRe50_DT))
plt.legend(loc=1)
plt.title("Apparent Error Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
precision_app_DT = PrecisionAtRe50_DT
print('Apparent Precision (Pr@Re50): ', precision_app_DT)

# #############################################################################
#   STEP 2: bootstrap  model on training data to get apparent errors
# #############################################################################
print('\n---- Estimate Error - Bootstrapping on Train Data ------------------')

y_scores = np.array([0.5])

# run bootstrap
stats_precision_DT = list()
for b_iter in range(n_iterations):
	# prepare train and test sets
    idxs = [j for j in range(len(train_data))]
    b_train_data, b_train_class, idx = resample(train_data, train_class, idxs, n_samples=n_size, stratify=train_class)
    test_idx = np.array([x for x in idxs if x not in idx])

    b_test_data = train_data.values[test_idx,:];
    b_test_class = train_class.values[test_idx];
    
    # **** BEGIN INSERT MODEL HERE **********************************************************
    # **** BEGIN INSERT MODEL HERE **********************************************************   
    dt = DecisionTreeClassifier(max_depth =  8, class_weight = {1: 5, 0: 1}, max_leaf_nodes=100)
    dt.fit(b_train_data, b_train_class)

    class_true = b_test_class
    class_pred = dt.predict(b_test_data).tolist()
    class_prob = dt.predict_proba(b_test_data)[::,1]

    precision, recall, thresholds = sklm.precision_recall_curve(class_true, class_prob)
    average_precision = sklm.average_precision_score(class_true, class_prob)

    # **** END INSERT MODEL HERE ************************************************************
    # **** END INSERT MODEL HERE ************************************************************
        
    PrecisionAtRe50_DT = np.max(precision[recall>=0.5])
    stats_precision_DT.append(PrecisionAtRe50_DT)
    print('Iter ', b_iter, ' Pr@Re50 = ', PrecisionAtRe50_DT)

# #############################################################################
# Bootstrapping Results 
print('\n----  Bootstrapping Results ----------------------------------------')
bootstrap_mean_DT = np.mean(stats_precision_DT)
bootstrap_std_DT = np.std(stats_precision_DT)
print('Precision Mean: ',bootstrap_mean_DT, ', Standard Deviation: ', bootstrap_std_DT)

# #############################################################################
#   STEP 3: Calculate 0.632 Bootstrapping value
# #############################################################################
print('\n---- 0.632 Bootstrapping value -------------------------------------')
bootstrap632_DT = 0.632*np.mean(stats_precision_DT) + 0.368*precision_app_DT
print('0.632 Bootstrapping Estimate = ', bootstrap632_DT)

# #############################################################################
#   STEP 4: Run Holdout Test on Test data
# #############################################################################
print('\n---- Holdout Test on Test Data -------------------------------------')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
prediction = dt.predict(test_data)
pred = dt.predict_proba(test_data)

# keep probabilities for the positive outcome only
pred = pred[:, 1]

precision, recall, thresholds = sklm.precision_recall_curve(test_class, pred)
average_precision = sklm.average_precision_score(test_class, pred)



PrecisionAtRe50_DT = np.max(precision[recall>=0.5])

plt.figure()
plt.plot(recall, precision,label="DecisionTreeClassifier, auc = {0:.4f}".format(average_precision))
plt.plot([0, 1], [PrecisionAtRe50_DT, PrecisionAtRe50_DT], '-r', linewidth=1, label="Pr@Re50 = {0:.4f}".format(PrecisionAtRe50_DT))
plt.legend(loc=1)
plt.title("Decision Tree Holdout Test Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

precision_test_DT = PrecisionAtRe50_DT
print('Test Precision (Pr@Re50): ', precision_test_DT)

# Plot standard Deviation of Bootstrap Test with Holdout Test Line and Apparent Error Line
plt.figure()
plt.hist(stats_precision_DT, 20, label='Bootstrap Precision', edgecolor='black')
ylim = plt.ylim()

plt.plot(2 * [precision_app_DT], ylim, '-g', linewidth=1, label="Apparent Pr@Re50 = {0:.4f}".format(precision_app_DT))
plt.plot(2 * [precision_test_DT], ylim, '-r', linewidth=1, label="Holdout Pr@Re50 = {0:.4f}".format(precision_test_DT))
plt.plot(2 * [bootstrap632_DT], ylim, '-m', linewidth=1, label="632 Bootstrap Pr@Re50 = {0:.4f}".format(bootstrap632_DT))
plt.plot(2 * [bootstrap_mean_DT], ylim, '-b', linewidth=1, label="Bootstrap Pr@Re50 = {0:.4f}".format(bootstrap_mean_DT))

plt.ylim(ylim)
plt.legend(loc=1)
plt.title("Decision Tree Precision Results")
plt.xlabel('Score')
plt.show()

# #############################################################################
#   STEP 5: Is classifier better than random? Permutation Test 
# #############################################################################
print('\n---- Permutation on Test Data --------------------------------------')
#https://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py
cv = StratifiedKFold(2)
score_DT, permutation_scores_DT, pvalue_DT = permutation_test_score(dt, test_data, test_class, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

print("Classification score %s (pvalue : %s)" % (score_DT, pvalue_DT))






# ================================================================================================================================================
# ================================================================================================================================================
# DECISION STUMP SECTION
# ================================================================================================================================================
# ================================================================================================================================================
print('\n=== DECISION STUMP =================================================')
# #############################################################################
#   STEP 1: run model on training data to get apparent errors
# #############################################################################
print('\n---- Apparent Error - Train on Training Data, Test on Test Data ----')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
data_train, data_verif, class_train, class_verif = train_test_split(train_data, train_class, test_size = 0.3, random_state = 2, stratify = train_class)

#data_verif, class_train, class_verif
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
clf.fit(data_train, ravel(class_train))
prediction = clf.predict(data_verif)
pred = clf.predict_proba(data_verif)

# keep probabilities for the positive outcome only
pred = pred[:, 1]

print ("Accuracy : " + str(sklm.accuracy_score(class_verif, prediction)*100))
precision, recall, thresholds = sklm.precision_recall_curve(class_verif, pred)
average_precision = sklm.average_precision_score(class_verif, pred)

# **** END INSERT MODEL HERE ************************************************************
# **** END INSERT MODEL HERE ************************************************************

PrecisionAtRe50_DS = np.max(precision[recall>=0.5])

plt.figure()
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))
plt.plot([0, 1], [PrecisionAtRe50_DS, PrecisionAtRe50_DS], '-r', linewidth=1, label="Pr@Re50 = {0:.4f}".format(PrecisionAtRe50_DS))
plt.legend(loc=1)
plt.title("Apparent Error Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
precision_app_DS = PrecisionAtRe50_DS
print('Apparent Precision (Pr@Re50): ', precision_app_DS)

# #############################################################################
#   STEP 2: bootstrap  model on training data to get apparent errors
# #############################################################################
print('\n---- Estimate Error - Bootstrapping on Train Data ------------------')

y_scores = np.array([0.5])

# run bootstrap
stats_precision_DS = list()
for b_iter in range(n_iterations):
	# prepare train and test sets
    idxs = [j for j in range(len(train_data))]
    b_train_data, b_train_class, idx = resample(train_data, train_class, idxs, n_samples=n_size, stratify=train_class)
    test_idx = np.array([x for x in idxs if x not in idx])

    b_test_data = train_data.values[test_idx,:];
    b_test_class = train_class.values[test_idx];

    # **** BEGIN INSERT MODEL HERE **********************************************************
    # **** BEGIN INSERT MODEL HERE **********************************************************   
    
	# evaluate model
    clfb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
    clfb.fit(b_train_data, ravel(b_train_class))
    prediction = clfb.predict(b_test_data)
    pred = clfb.predict_proba(b_test_data)
    
    # keep probabilities for the positive outcome only
    pred = pred[:, 1]
    
    precision, recall, thresholds = sklm.precision_recall_curve(b_test_class, pred)
    
    # **** END INSERT MODEL HERE ************************************************************
    # **** END INSERT MODEL HERE ************************************************************   PrecisionAtRe50_DS = np.max(precision[recall>=0.5])

    PrecisionAtRe50_DS = np.max(precision[recall>=0.5])
    stats_precision_DS.append(PrecisionAtRe50_DS)
    print('Iter ', b_iter, ' Pr@Re50 = ', PrecisionAtRe50_DS)

# #############################################################################
# Bootstrapping Results 
print('\n----  Bootstrapping Results ----------------------------------------')
bootstrap_mean_DS = np.mean(stats_precision_DS)
bootstrap_std_DS = np.std(stats_precision_DS)
print('Precision Mean: ',bootstrap_mean_DS, ', Standard Deviation: ', bootstrap_std_DS)

# #############################################################################
#   STEP 3: Calculate 0.632 Bootstrapping value
# #############################################################################
print('\n---- 0.632 Bootstrapping value -------------------------------------')
bootstrap632_DS = 0.632*np.mean(stats_precision_DS) + 0.368*precision_app_DS
print('0.632 Bootstrapping Estimate = ', bootstrap632_DS)

# #############################################################################
#   STEP 4: Run Holdout Test on Test data
# #############################################################################
print('\n---- Holdout Test on Test Data -------------------------------------')
# **** BEGIN INSERT MODEL HERE **********************************************************
# **** BEGIN INSERT MODEL HERE **********************************************************
prediction = clf.predict(test_data)
pred = clf.predict_proba(test_data)

# keep probabilities for the positive outcome only
pred = pred[:, 1]

precision, recall, thresholds = sklm.precision_recall_curve(test_class, pred)
average_precision = sklm.average_precision_score(test_class, pred)

# **** END INSERT MODEL HERE ************************************************************
# **** END INSERT MODEL HERE ************************************************************

PrecisionAtRe50_DS = np.max(precision[recall>=0.5])

plt.figure()
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))
plt.plot([0, 1], [PrecisionAtRe50_DS, PrecisionAtRe50_DS], '-r', linewidth=1, label="Pr@Re50 = {0:.4f}".format(PrecisionAtRe50_DS))
plt.legend(loc=1)
plt.title("Decision Stump Holdout Test Precision Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

precision_test_DS = PrecisionAtRe50_DS
print('Test Precision (Pr@Re50): ', precision_test_DS)

# Plot standard Deviation of Bootstrap Test with Holdout Test Line and Apparent Error Line
plt.figure()
plt.hist(stats_precision_DS, 20, label='Bootstrap Precision', edgecolor='black')
ylim = plt.ylim()

plt.plot(2 * [precision_app_DS], ylim, '-g', linewidth=1, label="Apparent Pr@Re50 = {0:.4f}".format(precision_app_DS))
plt.plot(2 * [precision_test_DS], ylim, '-r', linewidth=1, label="Holdout Pr@Re50 = {0:.4f}".format(precision_test_DS))
plt.plot(2 * [bootstrap632_DS], ylim, '-m', linewidth=1, label="632 Bootstrap Pr@Re50 = {0:.4f}".format(bootstrap632_DS))
plt.plot(2 * [bootstrap_mean_DS], ylim, '-b', linewidth=1, label="Bootstrap Pr@Re50 = {0:.4f}".format(bootstrap_mean_DS))

plt.ylim(ylim)
plt.legend(loc=1)
plt.title("Decision Stump Precision Results")
plt.xlabel('Precision')
plt.show()

# #############################################################################
#   STEP 5: Is classifier better than random? Permutation Test 
# #############################################################################
print('\n---- Permutation on Test Data --------------------------------------')
#https://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py
cv = StratifiedKFold(2)
score_DS, permutation_scores_DS, pvalue_DS = permutation_test_score(clf, test_data, test_class, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

print("Classification score %s (pvalue : %s)" % (score_DS, pvalue_DS))



# #############################################################################
# View histogram of permutation scores
plt.figure()
plt.hist(permutation_scores_DT, 20, label='Permutation scores', edgecolor='green')
plt.hist(permutation_scores_DS, 20, label='Permutation scores', edgecolor='blue')
ylim = plt.ylim()

plt.plot(2 * [score_DT], ylim, '--g', linewidth=1, label="Decision Tree Classification Score = {0:.4f}".format(pvalue_DT))
plt.plot(2 * [score_DS], ylim, '--b', linewidth=1, label="Decision Stump Classification Score = {0:.4f}".format(pvalue_DS))
plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()



print("Execution took %s seconds" % (time.time() - start_time))