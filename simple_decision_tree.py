import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import * 

# Import data
df = pd.read_csv("csv_result-Descriptors_Training.csv")

# Split into train and test
X = df.drop(['id', 'class'], axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

# Train and test model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_test) 

# Print results 
print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_pred)))

print ("Accuracy : " + str(accuracy_score(y_test,y_pred)*100))

print("Report : \n" + str(classification_report(y_test, y_pred)))
