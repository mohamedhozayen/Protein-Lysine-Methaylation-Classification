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
    return data[data.columns[selector.get_support(indices=True)]]
    
    

"""
    PCA one component
    assume df is standarized
    include column class in df
    n_c = 1 or 2
    
"""
def pca_linear(df, n_c=2, pos_only=False, plot = False):
    X = df.drop(['class'], axis=1)
    
    columns = []
    for i in range(1, n_c+1):
        columns.append('pca-'+str(i))
    
    pca = PCA(n_components=n_c)
    pca_result = pca.fit_transform(X)
    df_pca = pd.DataFrame(data = pca_result
                 , columns = columns)
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    df_pca = pd.concat([df_pca, df['class']], axis=1)
    #print(pca_comp)
    if n_c == 1:
        df_pca['pca-2'] = 0
        df_pca['pca-2'] = df_pca['class'][df_pca['class']==1]
        df_pca['pca-2'] = df_pca['pca-2'].fillna(0)
    
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if pos_only:
            df_pca.drop(df_pca[df_pca['class']==0].index, inplace=True)
            plt.figure(figsize=(16,7))
            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(
                    x="pca-1", y="pca-2",
                    palette=sns.color_palette("hls", 2),
                    data=df_pca,
                    legend="full",
                    alpha=0.6,
                    ax=ax1
                    )
        else:
    
            plt.figure(figsize=(16,7))
            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(
                    x="pca-1", y="pca-2",
                    hue="class",
                    palette=sns.color_palette("hls", 2),
                    data=df_pca,
                    legend="full",
                    alpha=0.6,
                    ax=ax1
                    )


    return pca, df_pca
    
    
    
    
    
    
    
    
    
    
    