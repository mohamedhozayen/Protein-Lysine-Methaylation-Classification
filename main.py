################################
# Main.py                      #
# SYSC 5405 - Term Project     #
# Group 7: Decision Trees      #
# By: Mo, Jason, Ian and Ben   #
################################
# Imports
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import preprocessing as prc
import feature_selection as fs
from random import seed
from random import randint
from sklearn.model_selection import *
from sklearn.tree import * 
from sklearn.metrics import * 
from sklearn.feature_selection import *
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
################################
startTime = datetime.now()
bootstrap_test_count = 10
rand_state = randint(0, 100)
################################
# Report will plot a PR curve and return the test stat
def report(name, y_true, y_pred, y_prob):
	cm = confusion_matrix(y_true, y_pred)
	precision, recall, thresholds = precision_recall_curve( y_true, y_prob)
	average_precision = average_precision_score(y_true, y_prob)
	PrecisionAtRe50_DT = np.max(precision[recall>=0.5])
	print("===== " + name +"=====")
	print("TN = " + str(cm[0][0]))
	print("FP = " + str(cm[0][1]))
	print("FN = " + str(cm[1][0]))
	print("TP = " + str(cm[1][1]))
	print("Pr = " + str(cm[1][1]/(cm[1][1] + cm[0][1])))
	print("Re = " + str(cm[1][1]/(cm[1][1] + cm[1][0])))
	print("Confusion Matrix: \n" + str(cm))
	print('Pr@Re50 = ', PrecisionAtRe50_DT)
	print()
	plt.plot(recall, precision,label=name + ", Pr@Re>50 = {0:.5f}".format(PrecisionAtRe50_DT))
	return PrecisionAtRe50_DT

# This function trains and tests a model
# Returns the predictions and their probabilty 
def test_model(model, X_train, X_test, y_train):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[::,1]
	return y_pred, y_prob

# Main function
# Inputs: model - will be trained and validated using k-fold
# [optional] clean_data or unsupervise_fs
def main(name, model, clean_data = False, unsupervise_fs = False):
	df = pd.read_csv('Files/csv_result-Descriptors_Training.csv', sep=',')
	df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])

	# Clean the data if the boolean is set.
	if clean_data:
		df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))

	if unsupervise_fs:
		fs_vairance = fs.variance_threshold(df, threshold=1)
		fs_vairance = pd.concat([fs_vairance, df['class']], axis=1)
	
	X = df.drop(['class'], axis=1)
	y = df['class']

	# Extract the subset for the holdout test
	# X, X_holdout, y, y_holdout = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify=y)
	
	y_pred = []
	y_prob = []
	y_true = []
	kf = StratifiedKFold(n_splits=5, shuffle = True, random_state = rand_state)
	for train_index, test_index in kf.split(X, y):
		# Split train and test set
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		# Train and test model  
		pred, prob = test_model(model, X_train, X_test, y_train)
		y_pred.extend(pred)
		y_prob.extend(prob)
		y_true.extend(y_test)

	return report(name, y_true, y_pred, y_prob)

# Evaluates the depth of the tree
def test_tree_depth():
	test_stats = {}
	for i in range(1, 15):
		dt = DecisionTreeClassifier(max_depth = i)
		test_stats[i] = main("DT with depth = " + str(i), dt)
	return test_stats

# Run the tree depth test
rslt = test_tree_depth()
print("Run Time: " + str(datetime.now() - startTime))

# Print PR Curves from test
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()

# Print out the distribution of curves 
plt.plot(list(rslt.keys()), list(rslt.values()))
plt.ylabel("Depth of Tree")
plt.xlabel("Pr@Re>50")
plt.title("Testing Decision Tree Depth")
plt.show()

# Test meta learning example
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
main("AdaBoost Decision Stumps", abc)
# Print PR Curves from test
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()