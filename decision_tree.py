import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.tree import * 
from sklearn.metrics import * 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import preprocessing as prc

# Import data
df = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.detect_outlier_iterative_IQR(df).fillna(0)

# Split into train and test
X = df.drop(['class'], axis=1)
y = df['class']

# NOTE Stratified KFold!
kf = StratifiedKFold(n_splits=5, shuffle = True)
kf.get_n_splits(X)


for i in range(1, 20):
	# for j in range (1,10):
	y_true = []
	y_pred = []
	y_prob = []
	for train_index, test_index in kf.split(X, y):
		# Train and test model  
		dt = DecisionTreeClassifier(max_depth = i, class_weight = {1: 5, 0: 1}, max_leaf_nodes=100)
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		dt.fit(X_train, y_train)

		y_true.extend(y_test)
		y_pred.extend(dt.predict(X_test).tolist())
		y_prob.extend(dt.predict_proba(X_test)[::,1])

	precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
	average_precision = average_precision_score(y_true, y_prob)
	plt.plot(recall, precision,label="Decision Tree, class weight: 5 to 1, depth {:d}, max_leaf_nodes = 100 auc={:.4f}".format(i, average_precision))

plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()
