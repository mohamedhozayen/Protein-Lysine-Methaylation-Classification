from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import preprocessing as prc

def report(y_true, y_pred, y_prob):
	cm = confusion_matrix(y_true, y_pred)
	print("TN = " + str(cm[0][0]))
	print("FP = " + str(cm[0][1]))
	print("FN = " + str(cm[1][0]))
	print("TP = " + str(cm[1][1]))
	print("Pr = " + str(cm[1][1]/(cm[1][1] + cm[0][1])))
	print("Re = " + str(cm[1][1]/(cm[1][1] + cm[1][0])))
	print("Confusion Matrix: \n" + str(cm))
	print()

	precision, recall, thresholds = precision_recall_curve( y_true, y_prob)
	average_precision = average_precision_score(y_true, y_prob)
	plt.plot(recall, precision,label=name+", auc = {0:.4f}".format(average_precision))

df = pd.read_csv('Files/csv_result-Descriptors_Training.csv', sep=',')
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))

dt = DecisionTreeClassifier()

X = df.drop(['class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify=y)

rfe = RFE(dt, 7)

X_train_rfe = rfe.fit_transform(X_train,y_train)
X_test_rfe = rfe.transform(X_test)

dt.fit(X_train_rfe,y_train)
y_pred = dt.predict(X_test_rfe)
y_prob = dt.predict_proba(X_test_rfe)

report(y_test, y_pred, y_prob)