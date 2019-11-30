# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:32:02 2019

@author: Mohamed Hozayen
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(x, y)
scores = selector.scores_

mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, columns):
    if bool:
        new_features.append(feature)
        print feature, round(scores[columns.index(feature)], 2)