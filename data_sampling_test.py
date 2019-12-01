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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import *
from sklearn import preprocessing
import preprocessing as prc
import feature_selection as fs


def report(y_true, y_pred, y_prob, name):
	print(name)
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
treeDepth = 2
# generate 2 class dataset
# Import data
df = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
# df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))

# Split into train and test
X = df.drop(['class'], axis=1)
y = df['class']
X_std = pd.DataFrame(preprocessing.normalize(X), columns=X.columns)
scaler = preprocessing.StandardScaler()
X_scl = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# NOTE Stratified KFold!
kf = StratifiedKFold(n_splits=5, shuffle = True)
kf.get_n_splits(X)
y_true = []

reg_dt = True
dt_y_pred = []
dt_y_prob = []

reg_dt_std = True
dt_std_y_pred = []
dt_std_y_prob = []

reg_dt_scl =True
dt_scl_y_pred = []
dt_scl_y_prob = []

rand = True
dc_y_pred = []
dc_y_prob = []

over_dt = True
dto_y_pred = []
dto_y_prob = []

under_dt = True
dtu_y_pred = []
dtu_y_prob = []

smote_dt = True
dts_y_pred = []
dts_y_prob = []

over_dt_scl = True
dto_scl_y_pred = []
dto_scl_y_prob = []

under_dt_scl = True
dtu_scl_y_pred = []
dtu_scl_y_prob = []

smote_dt_scl = True
dts_scl_y_pred = []
dts_scl_y_prob = []

clst_dt = False
dtc_y_pred = []
dtc_y_prob = []

for train_index, test_index in kf.split(X, y):
	# Train and test modelpip ion
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	X_train_std, X_test_std = X_std.iloc[train_index], X_std.iloc[test_index]
	X_train_scl, X_test_scl = X_scl.iloc[train_index], X_scl.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	y_true.extend(y_test)

	#Dummy classifier
	if rand:
		dc = DummyClassifier(strategy="stratified")
		dc.fit(X_train, y_train)
		dc_y_pred.extend(dc.predict(X_test).tolist())
		dc_y_prob.extend(dc.predict_proba(X_test)[::,1])

	# Clasic decision tree
	if reg_dt:
		dt = DecisionTreeClassifier(max_depth = treeDepth)
		dt.fit(X_train, y_train)
		dt_y_pred.extend(dt.predict(X_test).tolist())
		dt_y_prob.extend(dt.predict_proba(X_test)[::,1])

	# Clasic decision tree standardized data
	if reg_dt_std:
		dt_std = DecisionTreeClassifier(max_depth = treeDepth)
		dt_std.fit(X_train_std, y_train)
		dt_std_y_pred.extend(dt_std.predict(X_test_std).tolist())
		dt_std_y_prob.extend(dt_std.predict_proba(X_test_std)[::,1])
	
	# Clasic decision tree scaled data
	if reg_dt_scl:
		dt_scl = DecisionTreeClassifier(max_depth = treeDepth)
		dt_scl.fit(X_train_scl, y_train)
		dt_scl_y_pred.extend(dt_scl.predict(X_test_scl).tolist())
		dt_scl_y_prob.extend(dt_scl.predict_proba(X_test_scl)[::,1])

	# DT trained on oversampled data set
	if over_dt_scl:
		ros = RandomOverSampler()
		X_ros, y_ros = ros.fit_sample(X_train_scl, y_train)
		dto_scl = DecisionTreeClassifier()
		dto_scl.fit(X_ros, y_ros)
		dto_scl_y_pred.extend(dto_scl.predict(X_test_scl).tolist())
		dto_scl_y_prob.extend(dto_scl.predict_proba(X_test_scl)[::,1])

	# DT trained on SMOTE data set
	if smote_dt_scl:
		smote = SMOTE()
		X_smote, y_smote = smote.fit_sample(X_train_scl, y_train)
		dts_scl = DecisionTreeClassifier(max_depth = treeDepth)
		dts_scl.fit(X_smote, y_smote)
		dts_scl_y_pred.extend(dts_scl.predict(X_test_scl).tolist())
		dts_scl_y_prob.extend(dts_scl.predict_proba(X_test_scl)[::,1])

	# DT trained on upndersampled data set
	if under_dt_scl:
		rus = RandomUnderSampler(return_indices=True)
		X_rus, y_rus, id_rus = rus.fit_sample(X_train_scl, y_train)
		dtu_scl = DecisionTreeClassifier(max_depth = treeDepth)
		dtu_scl.fit(X_rus, y_rus)
		dtu_scl_y_pred.extend(dtu_scl.predict(X_test_scl).tolist())
		dtu_scl_y_prob.extend(dtu_scl.predict_proba(X_test_scl)[::,1])

	# DT trained on oversampled data set
	if over_dt:
		ros = RandomOverSampler()
		X_ros, y_ros = ros.fit_sample(X_train, y_train)
		dto = DecisionTreeClassifier()
		dto.fit(X_ros, y_ros)
		dto_y_pred.extend(dto.predict(X_test).tolist())
		dto_y_prob.extend(dto.predict_proba(X_test)[::,1])

	# DT trained on SMOTE data set
	if smote_dt:
		smote = SMOTE()
		X_smote, y_smote = smote.fit_sample(X_train, y_train)
		dts = DecisionTreeClassifier(max_depth = treeDepth)
		dts.fit(X_smote, y_smote)
		dts_y_pred.extend(dts.predict(X_test).tolist())
		dts_y_prob.extend(dts.predict_proba(X_test)[::,1])

	# DT trained on upndersampled data set
	if under_dt:
		rus = RandomUnderSampler(return_indices=True)
		X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)
		dtu = DecisionTreeClassifier(max_depth = treeDepth)
		dtu.fit(X_rus, y_rus)
		dtu_y_pred.extend(dtu.predict(X_test).tolist())
		dtu_y_prob.extend(dtu.predict_proba(X_test)[::,1])

	# DT trained on ClusterCentroids data set
	if clst_dt:
		cc = ClusterCentroids()
		X_cc, y_cc = cc.fit_sample(X_train, y_train)
		dtc = DecisionTreeClassifier(max_depth = treeDepth)
		dtc.fit(X_cc, y_cc)
		dtc_y_pred.extend(dtc.predict(X_test).tolist())
		dtc_y_prob.extend(dtc.predict_proba(X_test)[::,1])

if reg_dt:
	report(y_true, dt_y_pred, dt_y_prob, "Decision Tree Trained on Unbalanced Data")

if reg_dt_std:
	report(y_true, dt_std_y_pred, dt_std_y_prob, "Decision Tree Trained on on Standardized Unbalanced Data")

if reg_dt_scl:
	report(y_true, dt_scl_y_pred, dt_scl_y_prob, "Decision Tree Trained on Scaled Unbalanced Data")

if smote_dt_scl:
	report(y_true, dts_scl_y_pred, dts_scl_y_prob, "Decision Tree Trained on Scaled SMOTE Data")

if under_dt_scl:
	report(y_true, dtu_scl_y_pred, dtu_scl_y_prob, "Decision Tree Trained on Scaled UnderSampled Data")

if over_dt_scl:
	report(y_true, dto_scl_y_pred, dto_scl_y_prob, "Decision Tree Trained on Scaled Oversampled Data")

if smote_dt:
	report(y_true, dts_y_pred, dts_y_prob, "Decision Tree Trained on SMOTE Data")

if under_dt:
	report(y_true, dtu_y_pred, dtu_y_prob, "Decision Tree Trained on UnderSampled Data")

if over_dt:
	report(y_true, dto_y_pred, dto_y_prob, "Decision Tree Trained on Oversampled Data")

if clst_dt:
	report(y_true, dtc_y_pred, dtc_y_prob, "Decision Tree Trained on ClusterCentroids Data")

if rand:
	report(y_true, dc_y_pred, dc_y_prob, "Random Classifier")


plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()

# print ("Accuracy : " + str(accuracy_score(y_true,dt_y_pred)*100))
# print("Report : \n\n" + str(classification_report(y_true, dt_y_pred)))