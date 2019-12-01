# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:27:46 2019

 features_optimal.py                      
 SYSC 5405 - Term Project     
 Group 7: Decision Trees      
 By: Mo, Jason, Ian and Ben 

@author: Mohamed Hozayen
"""

import numpy as np
import pandas as pd
import preprocessing as prc
import feature_selection as fs
import main


# optional: use to vote for best depth - insanity!!
def main_best_n(data, n): 
    l=[]
    for i in range(0,n):
        rslt = main.test_tree_depth(data)
        l.append([rslt.index(max(rslt)), max(rslt)])
    return l

df = pd.read_csv('Files\csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df))
df = prc.standarize(df) # or normalize

#summary = []
#summary_balance = []

pca_rbf = fs.pca_kernel(df, kernel='rbf')
pca_poly = fs.pca_kernel(df, kernel='poly') 
pca_cos = fs.pca_kernel(df, kernel='cosine')

pca_rbf.to_csv('pca-rbf-features.csv')
pca_poly.to_csv('pca-poly-features.csv')
pca_cos.to_csv('pca-cos-features.csv')

#summary_table_balance = pd.DataFrame(summary_balance)
#summary_table_balance.columns = ['method-balance', 'optimal tree depth', 'pre@recall50']
#summary_table_balance.to_csv('unsupervised-features-balance-summary_table.csv')


"""
vt = fs.variance_threshold(df, threshold=1)
rslt_vt = main.test_tree_depth(vt, class_weight="balanced")
summary_balance.append(['variance-threshold', rslt_vt.index(max(rslt_vt)), max(rslt_vt)])

pca = fs.pca_linear(df, n=2) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-2', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=7) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-7', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=9) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-9', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=10) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-10', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=15) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-15', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

tsne = fs.tsne(df, n=1) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-1', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])   

tsne = fs.tsne(df, n=2) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-2', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

tsne = fs.tsne(df, n=3) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-3', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

pca_kernel = fs.pca_kernel(df, kernel='rbf') #kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['rbf-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='poly') 
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['poly-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='cosine')
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['cosine-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

"""
