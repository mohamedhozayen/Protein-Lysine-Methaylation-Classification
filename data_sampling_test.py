# pip3 install imblearn

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.tree import *
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn.tree import *
from sklearn import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# generate 2 class dataset
# Import data
df = pd.read_csv("csv_result-Descriptors_Training.csv")

# Split into train and test
X = df.drop(['id', 'class'], axis=1)
y = df['class']

# NOTE Stratified KFold!
kf = StratifiedKFold(n_splits=5, shuffle = True)
kf.get_n_splits(X)
y_true = []
dt_y_pred = []
dt_y_prob = []
dc_y_pred = []
dc_y_prob = []
dto_y_pred = []
dto_y_prob = []
dtu_y_pred = []
dtu_y_prob = []

for train_index, test_index in kf.split(X, y):
	# Train and test modelpip ion
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	# y_true.extend(y_test == 'P')
	y_true.extend(y_test)
	smote = SMOTE()
	X_smote, y_smote = smote.fit_sample(X_train, y_train)
	rus = RandomUnderSampler(return_indices=True)
	X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)

	#Dummy classifier
	dc = DummyClassifier(strategy="stratified")
	dc.fit(X_train, y_train)
	dc_y_pred.extend(dc.predict(X_test).tolist())
	dc_y_prob.extend(dc.predict_proba(X_test)[::,1])

	# Clasic decision tree
	dt = DecisionTreeClassifier()
	dt.fit(X_train, y_train)
	dt_y_pred.extend(dt.predict(X_test).tolist())
	dt_y_prob.extend(dt.predict_proba(X_test)[::,1])

	# DT trained on oversampled data set
	dto = DecisionTreeClassifier()
	dto.fit(X_smote, y_smote)
	dto_y_pred.extend(dto.predict(X_test).tolist())
	dto_y_prob.extend(dto.predict_proba(X_test)[::,1])

	# DT trained on upndersampled data set
	dtu = DecisionTreeClassifier()
	dtu.fit(X_rus, y_rus)
	dtu_y_pred.extend(dtu.predict(X_test).tolist())
	dtu_y_prob.extend(dtu.predict_proba(X_test)[::,1])

print("Decision Tree Trained on Unbalanced Data:")
print("Confusion Matrix: \n" + str(confusion_matrix(y_true, dt_y_pred)))
print ("Accuracy : " + str(accuracy_score(y_true,dt_y_pred)*100))
print("Report : \n\n" + str(classification_report(y_true, dt_y_pred)))

print("Decision Tree Trained on Oversampled Data:")
print("Confusion Matrix: \n" + str(confusion_matrix(y_true, dtu_y_pred)))
print ("Accuracy : " + str(accuracy_score(y_true,dtu_y_pred)*100))
print("Report : \n" + str(classification_report(y_true, dtu_y_pred)))

print("Decision Tree Trained on UnderSampled Data:")
print("Confusion Matrix: \n" + str(confusion_matrix(y_true, dto_y_pred)))
print ("Accuracy : " + str(accuracy_score(y_true,dto_y_pred)*100))
print("Report : \n" + str(classification_report(y_true, dto_y_pred)))

print("Random Classifier:")
print("Confusion Matrix: \n" + str(confusion_matrix(y_true, dc_y_pred)))
print ("Accuracy : " + str(accuracy_score(y_true,dc_y_pred)*100))
print("Report : \n" + str(classification_report(y_true, dc_y_pred)))



# precision, recall, thresholds = precision_recall_curve( y_true, dtu_y_prob)
# average_precision = average_precision_score(y_true, dtu_y_prob)
# plt.plot(recall, precision,label="DecisionTree With Under Sampled Training Data, auc = {0:.4f}".format(average_precision))

# precision, recall, thresholds = precision_recall_curve( y_true, dto_y_prob)
# average_precision = average_precision_score(y_true, dto_y_prob)
# plt.plot(recall, precision,label="DecisionTree With Over Sampled Training Data, auc = {0:.4f}".format(average_precision))

# precision, recall, thresholds = precision_recall_curve( y_true, dt_y_prob)
# average_precision = average_precision_score(y_true, dt_y_prob)
# plt.plot(recall, precision,label="DecisionTreeClassifier, auc = {0:.4f}".format(average_precision))

# precision, recall, thresholds = precision_recall_curve(y_true, dc_y_prob)
# average_precision = average_precision_score(y_true, dc_y_prob)
# plt.plot(recall, precision,label="DummyClassifier, auc={0:.4f}".format(average_precision))
# plt.legend(loc=1)
# plt.title("Precision Recall Curve")
# plt.show()
