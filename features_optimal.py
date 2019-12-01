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
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))
df = prc.standarize(df) # or normalize

summary = []

vt = fs.variance_threshold(df, threshold=1)
rslt_vt = main.test_tree_depth(vt)
summary.append(['variance-threshold', rslt_vt.index(max(rslt_vt)), max(rslt_vt)])

pca = fs.pca_linear(df, n=10) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca)
summary.append(['pca-10', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca_kernel = fs.pca_kernel(df, kernel='rbf') #kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
rslt_kernel = main.test_tree_depth(pca_kernel)
summary.append(['rbf-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

tsne = fs.tsne(df, n=1) #3 is max
rslt_tsne = main.test_tree_depth(tsne)
summary.append(['tsne-1', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])   

pca_kernel = fs.pca_kernel(df, kernel='poly') 
rslt_kernel = main.test_tree_depth(pca_kernel)
summary.append(['poly-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='cosine')
rslt_kernel = main.test_tree_depth(pca_kernel)
summary.append(['cosine-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])



"""
try after: DT param class_weight="auto" or "balanced" 
"""
summary_table = pd.DataFrame(summary)
summary_table.columns = ['method', 'optimal tree depth', 'pre@recall50']
summary_table.to_csv('unsupervised-features-summary_table.csv')






















