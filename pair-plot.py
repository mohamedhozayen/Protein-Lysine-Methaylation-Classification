# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:12:31 2019

@author: Mohamed Hozayen
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.tree import *
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn.tree import *
from sklearn import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import *
from sklearn import preprocessing
import preprocessing as prc
import feature_selection as fs

df = pd.read_csv('Files\csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))
df_norm = prc.normalize(df) #normalize

vars_variance = ['IP_ES_25_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1',	
        'Gs(U)_IB_68_N1', 'Z1_NO_sideR35_CV',
        'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71',
        'Z3_NO_UCR_S', 'Pb_NO_PCR_V']

vars_cluster = ['Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1',
         'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'class']

df_sub = df[vars2].drop(['class'], axis=1)
df_sub = np.exp(df_sub)
df_sub = pd.concat([df_sub, df['class']], axis=1)

df_p = df_sub[df_sub['class']==1]
df_n = df_sub[df_sub['class']==0]

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
f_anova = fs.select_k_best(df.iloc[:,:-1], df.iloc[:,-1], f_classif, 7)
df_sub = df[f_anova.iloc[:,0].append(pd.Series('class'))] #
df_p = df_sub[df_sub['class']==1]
df_n = df_sub[df_sub['class']==0]

spearman = fs.corr_linear(df.iloc[:,:-1], df.iloc[:,-1], method='spearman')
df_sub = df[spearman.iloc[:10,0].append(pd.Series('class'))] #
df_p = df_sub[df_sub['class']==1]
df_n = df_sub[df_sub['class']==0]

import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(df_sub, hue="class")
g.set(yscale="log")

g = sns.pairplot(df_p, hue="class", palette="husl")
g = sns.pairplot(df_n, hue="class", markers=["o"])







