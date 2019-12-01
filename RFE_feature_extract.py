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

def report(name, y_true, y_pred, y_prob):
	cm = confusion_matrix(y_true, y_pred)
	precision, recall, thresholds = precision_recall_curve( y_true, y_prob)
	average_precision = average_precision_score(y_true, y_prob)
	PrecisionAtRe50_DT = np.max(precision[recall>=0.5])
	print("TN = " + str(cm[0][0]))
	print("FP = " + str(cm[0][1]))
	print("FN = " + str(cm[1][0]))
	print("TP = " + str(cm[1][1]))
	print("Pr = " + str(cm[1][1]/(cm[1][1] + cm[0][1])))
	print("Re = " + str(cm[1][1]/(cm[1][1] + cm[1][0])))
	print("Confusion Matrix: \n" + str(cm))
	print()

	print('Pr@Re50 = ', PrecisionAtRe50_DT)
	plt.plot(recall, precision,label="Pr@Re>50 = {0:.5f}".format(PrecisionAtRe50_DT))

def test_model(model, X_train, X_test, y_train):
	model.fit(X_train, y_train)
	y_pred = dt_rfe.predict(X_test_rfe)
	y_prob = dt_rfe.predict_proba(X_test_rfe)[::,1]
	return y_pred, y_prob


df = pd.read_csv('Files/csv_result-Descriptors_Training.csv', sep=',')
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
# df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))

X = df.drop(['class'], axis=1)
y = df['class']

# Split data for simple test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify=y)

# Do RFE
rfe = RFE(dt_rfe, 7)
X_train_rfe = rfe.fit_transform(X_train,y_train)
X_test_rfe = rfe.transform(X_test)
print("RFE feature ranking:")
print(rfe.ranking_)
print()

# Train and test classifier with RFE
dt_rfe = DecisionTreeClassifier(max_depth=4)
y_pred, y_prob = test_model(dt_rfe, X_train_rfe, y_train, X_test_rfe)
report("RFE: ", y_test, y_pred, y_prob)

# Train and test classifier without RFE
dt = DecisionTreeClassifier(max_depth=4)
y_pred, y_prob = test_model(dt_rfe, X_train_rfe, y_train, X_test_rfe)
report("RFE: ", y_test, dt.predict(X_test), dt.predict_proba(X_test)[::,1])

# Display PR Curve
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()