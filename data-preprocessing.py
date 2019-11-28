"""
Created on Wed Nov 27 18:44:07 2019

Data Preprocessing Template

@author: mohamedhozayen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

#Figures = 'Figures/'
#Files = 'Files/'
#val = pd.read_csv('csv_result-Descriptors_Calibration.csv', sep=',')

train = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
train = train.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])

#negative_index = train[train['class'] == 0 ].index

t_feature_positive = train[train['class'] == 1]
t_feature_positive = t_feature_positive.drop(['class'], axis=1)

t_feature_negative = train[train['class'] == 0]
t_feature_negative = t_feature_negative.drop(['class'], axis=1)

feature_negative_stats = t_feature_negative.describe()
feature_positive_stats = t_feature_positive.describe()

preprocess_negative = train[train['class'] == 0]
preprocess_negative = preprocess_negative.drop(['class'], axis=1)

preprocess_negative_stats_before = preprocess_negative.describe()
m_b = median_stat(feature_positive_stats, preprocess_negative.describe())
cutoff = 500
preprocess_negative = cut_threshold(preprocess_negative, cutoff) 
m_a = median_stat(feature_positive_stats, preprocess_negative.describe())

preprocess_negative = handle_outlier(preprocess_negative)

# replace outliers with -1. 
# outliers outside mean +/- 1.5*iqr
# use imputation to replace -1
def handle_outlier(og_df):
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=-1, strategy='mean')
    df = og_df.copy()
    stat = df.describe()
    for column in df.columns:
    #    for element in t_feature[column]:
        median = stat[column]['50%']
        min25 = stat[column]['25%']
        max75 = stat[column]['75%']
        mean = stat[column]['mean']
        std = stat[column]['std']
        iqr = (max75-min25)*1.5
        max_val = stat[column]['max']
        min_val = stat[column]['min']
        upper_bound = mean + 1.5*iqr
        lower_bound = mean - 1.5*iqr
        df[column].mask(df[column] > upper_bound, -1)
        df[column].mask(df[column] < lower_bound, -1)
#        and df[column] < lower_bound
#        df[column] = df[column].apply(lambda x: [y if y < upper_bound else -1 for y in x])
#        df[column] = df[column].apply(lambda x: [y if y > lower_bound else -1 for y in x])
    return df
#        preprocess_negative = preprocess_negative[preprocess_negative[column] < mean + max75]
#        preprocess_negative = preprocess_negative[preprocess_negative[column] > mean - min25]
#    preprocess_negative = preprocess_negative[preprocess_negative[column] < (median + iqr)]
#    preprocess_negative = preprocess_negative[preprocess_negative[column] > (median - iqr)]
#    preprocess_negative = preprocess_negative[preprocess_negative[column] < (median + 3.0*std)]
#    preprocess_negative = preprocess_negative[preprocess_negative[column] > (median - 3.0*std)]
#           median = stat[column]['50%']
#        min25 = stat[column]['25%']
#        max75 = stat[column]['75%']
#        mean = stat[column]['mean']
#        std = stat[column]['std']
#        iqr = (max75-min25)*1.5
#        max_val = stat[column]['max']
#        min_val = stat[column]['min'] 

preprocess_negative_stats_after = preprocess_negative.describe()


def median_stat(df_positive_stat, df_negative_stat):
    median50 = []
    for c in df_positive_stat.columns:
        median50.append([c, df_positive_stat[c]['50%'], df_negative_stat[c]['50%']])
    median50 = pd.DataFrame(median50)
    median50.columns = ['features', 'median Positive', 'median negative']
    return median50


def drop(df, cuttoff):
    for column in df.columns:
        df[column] = df[column].apply(lambda x: [y if y <= 9 else 11 for y in x])

for column in t_feature_negative.columns:
    check = preprocess_negative[column].between(-100, 100)

def cut_threshold(og_df, t):
    df = og_df.copy()
    for column in df.columns:
        i_high = df[df[column] > t].index
        i_low = df[df[column] < -t].index
        df.drop(i_high, inplace=True)
        df.drop(i_low, inplace=True)
    return df
#    [x for x in arr if (x > mean - 2 * sd)]
    
    
    
#    final_list = [x for x in arr if (x > mean - 2 * sd)]



    
#t_feature.iloc[:,0]
#len([i for i in t_feature[t_feature.columns[1]] if i > 100])

#from sklearn.cluster import DBSCAN
#outlier_detection = DBSCAN(
#  eps = 0.5,
#  metric="euclidean",
#  min_samples = 20,
#  n_jobs = -1)
#clusters = outlier_detection.fit_predict(l)
#
#from matplotlib import cm
#cmap = cm.get_cmap('Accent')
#l.plot.scatter(
#  x = t_feature.columns[1],
#  y = "class",
#  c = clusters,
#  cmap = cmap
#)


## Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""
#
#"""
## Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])
#"""

