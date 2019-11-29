#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:55:36 2019

@author: mohamedhozayen
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


"""
method = 
f_classif
        ANOVA F-value between label/feature for classification tasks.    
"""
def select_k_best(features, target, method, k=10):
    
    selector = SelectKBest(method, k)
    selector.fit_transform(features, target)
    scores = selector.scores_
    
    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features
    columns = features.columns
    for bool, feature in zip(mask, columns):
        if bool:
            new_features.append([feature, 
                                round(scores[columns.get_loc(feature)])])
    
    new_features = pd.DataFrame(new_features)
    new_features.columns = ['feature', 'score']
    new_features = new_features.sort_values(by=['score'], ascending = False)
    return new_features

def  corr_linear(features, target, method='spearman'):   
    l = []
    for column in features:
        l.append([column, features[column].corr(target, method)])

    scores = pd.DataFrame(l)
    scores.columns = ['feature', 'score']
    scores = scores.sort_values(by=['score'], ascending = False)
    return scores
