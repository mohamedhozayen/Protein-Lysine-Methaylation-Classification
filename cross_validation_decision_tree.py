import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn.tree import *
from sklearn import *
from pandas_ml import ConfusionMatrix

# Import data
df = pd.concat([pd.read_csv("csv_result-Descriptors_Training.csv"), pd.read_csv("csv_result-Descriptors_Calibration.csv")], axis=0, ignore_index=True)
# Split into train and test
X = df.drop(['class'], axis=1)
y = df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

# Cross validate:
kf = KFold(n_splits=5, shuffle = True, random_state = 12)
kf.get_n_splits(X)
y_true = []
y_pred = []
y_prob = []

for train_index, test_index in kf.split(X):
	# Train and test model
	dt = DecisionTreeClassifier()
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	dt.fit(X_train, y_train)

	y_true.extend(y_test)
	y_pred.extend(dt.predict(X_test).tolist())
	y_prob.extend(dt.predict_proba(X_test)[::,1])

print("Confusion Matrix: \n" + str(confusion_matrix(y_true, y_pred)))

print ("Accuracy : " + str(accuracy_score(y_true,y_pred)*100))

print("Report : \n" + str(classification_report(y_true, y_pred)))

# cm = ConfusionMatrix(y_true, y_pred)
# cm.print_stats()
# sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='g')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title("Confusion Matrix (All Features)")
# plt.show()

# auc = metrics.roc_auc_score(y_true, y_prob)
# fpr, tpr, _ = metrics.roc_curve(y_true,  y_prob)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.title("ROC Curve (All Features)")
# plt.show()

# precision, recall, thresholds = precision_recall_curve( y_true, y_prob)
# average_precision = average_precision_score(y_true, y_prob)
# plt.step(recall, precision,label="data 1, auc="+str(average_precision))
# plt.legend(loc=4)
# plt.title("Precision Recall Curve (All Features)")
# plt.show()
