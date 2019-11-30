#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:55:36 2019

@author: mohamedhozayen
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA


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

"""
    PCA one component
    assume df is standarized
    include column class in df
    n_c = 1 or 2
    
"""
def pca_linear(df, n_c=2):
    X = df.drop(['class'], axis=1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df_pca = pd.DataFrame(data = pca_result
                 , columns = ['pca-1', 'pca-2'])
    
#    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    
    #print(pca_comp)
    if n_c == 1:
        df_pca['pca-2'] = 0
    df_pca = pd.concat([df_pca, df['class']], axis=1)
    
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    return pca
    