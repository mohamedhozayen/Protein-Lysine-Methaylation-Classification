import pandas as pd
import seaborn as sns
import collections
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.tree import DecisionTreeClassifier
from numpy import ravel
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier


raw_data_df = pd.read_csv("train_data.csv")
raw_class_df = pd.read_csv("train_class.csv")

start_time = time.time()
data_train, data_verif, class_train, class_verif = train_test_split(raw_data_df, 
                                                                    raw_class_df, 
                                                                    test_size = 0.3, 
                                                                    random_state = 2, 
                                                                    stratify = raw_class_df)

#data_verif, class_train, class_verif
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
clf.fit(data_train, ravel(class_train))

prediction = clf.predict(data_verif)

pred = clf.predict_proba(data_verif)

tn, fp, fn, tp = confusion_matrix(class_verif, prediction).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: ", tp)
print("Confusion Matrix: \n" + str(confusion_matrix(class_verif, prediction)))
print ("Accuracy : " + str(accuracy_score(class_verif, prediction)*100))
print("Report : \n" + str(classification_report(class_verif, prediction)))

# keep probabilities for the positive outcome only
pred = pred[:, 1]

#print(np.shape(prediction))
#print(class_verif)

precision, recall, thresholds = precision_recall_curve(class_verif, pred, pos_label="P")
average_precision = average_precision_score(class_verif, pred, pos_label="P")
plt.plot(recall, precision,label="DecisionStumpClassifier, auc = {0:.4f}".format(average_precision))

plt.legend(loc=1)
plt.title("Precision Recall Curve")
plt.show()

print("Execution took %s seconds" % (time.time() - start_time))
