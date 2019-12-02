#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:55:36 2019


https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/
https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/

@author: mohamedhozayen
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold, RFE, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.tree import *

"""
    select_k_best_ANOVA uses f_classif (ANOVA analysis of variance)
            ANOVA F-value between label/feature for classification tasks.    
            
            note that anova assumes:
                1. The samples are independent
                2. Each sample is from a normally distributed population
                3. The population standard deviations of the groups are all equal. This
                property is known as homoscedasticity.
        
    choose best k features
    
    return data with features ranked from highest to lowest score

"""
def select_k_best_ANOVA(df, k=10):
    features = df.drop(['class'], axis=1)
    target = df['class']
    
    selector = SelectKBest(f_classif, k)
    selector.fit_transform(features, target)
    scores = selector.scores_
    
    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features
    columns = features.columns
    for bool, feature in zip(mask, columns):
        if bool:
            new_features.append([feature, round(scores[columns.get_loc(feature)])])

    new_features = pd.DataFrame(new_features)
    new_features.columns = ['feature', 'score']
    new_features = new_features.sort_values(by=['score'], ascending = False)
    top_score = df[new_features['feature']]
    top_score = pd.concat([top_score, df['class']], axis=1)
    return top_score

"""
    corr_linear performs pearson and spearman correlation             
            note that pearson and spearman assumes:
                1. The samples are independent
    
    return data with features ranked from highest to lowest score
"""
def  corr_linear(df, method='spearman'):
    features = df.drop(['class'], axis=1)
    target = df['class']
    c = []
    for column in features:
        c.append([column, features[column].corr(target, method)])

    scores = pd.DataFrame(c)
    scores.columns = ['feature', 'score']
    scores['score'] = abs(scores['score'])
    scores = scores.sort_values(by=['score'], ascending = False)
    top_score = df[scores['feature']]
    top_score = pd.concat([top_score, df['class']], axis=1)
    return top_score

"""
    RFE_DT | Recursive feature elimination using dicsion trees
    
    return data with ranked features
"""
def RFE_DT(df, test_size=0.3, n_features=7, max_depth=4):
    X = df.drop(['class'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 1, stratify=y)
    dt_rfe = DecisionTreeClassifier(max_depth=max_depth)
    rfe = RFE(dt_rfe, n_features)
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
 
    columns = X.columns
    rank = pd.DataFrame({'feature': columns, 'rank': list(rfe.ranking_)})
    rank = rank.sort_values(by=['rank'], ascending = True)
    top_rank = df[rank['feature'][rank['rank']==1]]
    top_rank = pd.concat([top_rank, df['class']], axis=1)
    return top_rank

"""
    RFECV_DT | Recursive feature elimination with cross validation using dicsion trees
    
    return data with ranked features
"""
def RFECV_DT(df, test_size=0.3, cv=5, min_features_to_select=7, max_depth=4):
    X = df.drop(['class'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 1, stratify=y)
    dt_rfecv = DecisionTreeClassifier(max_depth=max_depth)
    rfecv = RFECV(dt_rfecv, min_features_to_select=min_features_to_select, cv=StratifiedKFold(cv))
    X_train_rfecv = rfecv.fit_transform(X_train,y_train)
    X_test_rfecv = rfecv.transform(X_test)
 
    columns = X.columns
    rank = pd.DataFrame({'feature': columns, 'rank': list(rfecv.ranking_)})
    rank = rank.sort_values(by=['rank'], ascending = True)
    top_rank = df[rank['feature'][rank['rank']==1]]
    top_rank = pd.concat([top_rank, df['class']], axis=1)
    return top_rank


"""
    Variance threshold
    return data with high variance
"""
def variance_threshold(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    vt = data[data.columns[selector.get_support(indices=True)]]
    vt = pd.concat([vt, data['class']], axis=1)
    return vt
    

"""
    PCA linear | Principal Component Analysis 
    return data with projected features onto new dimension
"""
def pca_linear(df, n=2):
    X = df.drop(['class'], axis=1)

    pca = PCA(n_components = n)
    pca_result = pca.fit_transform(X)
    
    columns = []
    for i in range(1, len(pca_result[0])+1):
        columns.append('pca-'+str(i))
    df_pca = pd.DataFrame(data = pca_result, columns = columns)  
    df_pca = pd.concat([df_pca, df['class']], axis=1)
    return df_pca

"""
    PCA kernel |  Principal Component Analysis 
    kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
    
    return data with projected features onto new dimension using kernel mode
"""    

def pca_kernel(df, kernel='rbf'):
    if 'class' in df:
        X = df.drop(['class'], axis=1)
    else:
        X = df
   
    kpca = KernelPCA(kernel=kernel, fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    
    columns = []
    for i in range(1, len(X_back[0])+1):
        columns.append('pca-'+ kernel +str(i))
    X_back = pd.DataFrame(data = X_back, columns = columns)
    
    if 'class' in df:
        new = pd.concat([X_back, df['class']], axis=1)
    else:
        new = X_back
    
    return new

"""
    tSNE | t-distributed Stochastic Neighbor Embedding
    
    return data with projected features onto new dimension using kernel mode
"""    
def tsne(df, n=2):
    X = df.drop(['class'], axis=1)
    tsne = TSNE(n_components=n, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    
    columns = []
    for i in range(1, len(tsne_results[0])+1):
        columns.append('tsne-' + str(i))
    df_tsne = pd.DataFrame(data = tsne_results, columns = columns)
    df_tsne = pd.concat([df_tsne, df['class']], axis=1)
    return df_tsne
    
    
"""
    PCA linear for big data otherwise use pca_linear
    not finished!!
"""
def pca_incremental(df, n_c=7):
    X = df.drop(['class'], axis=1)
    transformer = IncrementalPCA(n_components=7)
    X_transformed = transformer.fit_transform(X)
    return X_transformed    