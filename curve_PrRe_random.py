# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.tree import *
from sklearn.dummy import DummyClassifier
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
# generate 2 class dataset

# Import data
df = pd.read_csv("csv_result-Descriptors_Training.csv")

# Split into train and test
X = df.drop(['id', 'class'], axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
y_test = y_test == 'P'

# generate a no skill prediction (majority class)
# fit a model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 

dc = DummyClassifier(strategy="stratified")
dc.fit(X_train, y_train)

# predict probabilities
dt_prob = dt.predict_proba(X_test)
dc_prob = dc.predict_proba(X_test)

# keep probabilities for the positive outcome only
dt_prob = dt_prob[:, 1]
dc_prob = dc_prob[:, 1]

precision, recall, thresholds = precision_recall_curve( y_test, dt_prob)
average_precision = average_precision_score(y_test, dt_prob)
plt.plot(recall, precision,label="DecisionTreeClassifier, auc = {0:.4f}".format(average_precision))


precision, recall, thresholds = precision_recall_curve( y_test, dc_prob)
average_precision = average_precision_score(y_test, dc_prob)
plt.plot(recall, precision,label="DummyClassifier, auc={0:.4f}, Re@Pr50={0:.4f}".format(average_precision))
plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()
