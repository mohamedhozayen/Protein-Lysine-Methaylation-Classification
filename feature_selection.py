#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:55:36 2019


https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/
https://stackabuse.com/applying-wrapper-methods-in-python-for-feature-selection/

@author: mohamedhozayen
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.manifold import TSNE

"""
method = 
f_classif
        ANOVA F-value between label/feature for classification tasks.    
        
        note that anova assumes:
            1. The samples are independent
            2. Each sample is from a normally distributed population
            3. The population standard deviations of the groups are all equal. This
            property is known as homoscedasticity.
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


    

def pca_incremental(df, n_c=7):
    X = df.drop(['class'], axis=1)
    transformer = IncrementalPCA(n_components=7)
    X_transformed = transformer.fit_transform(X)
    return X_transformed
    
"""
    return all features with at least thershold 
    no selection below 1 !!!
"""
def variance_threshold(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    vt = data[data.columns[selector.get_support(indices=True)]]
    vt = pd.concat([vt, data['class']], axis=1)
    return vt
    

"""
    PCA one component
    assume df is standarized
    include column class in df
    n_c = 1 or 2
    
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
    
#kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
def pca_kernel(df, kernel='rbf'):
    X = df.drop(['class'], axis=1)
   
    kpca = KernelPCA(kernel=kernel, gamma=10)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    
    columns = []
    for i in range(1, len(X_back[0])+1):
        columns.append('pca-'+ kernel +str(i))
    X_back = pd.DataFrame(data = X_back, columns = columns)
    new = pd.concat([X_back, df['class']], axis=1)
    return new

    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    