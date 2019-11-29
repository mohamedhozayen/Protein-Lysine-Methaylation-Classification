"""
Created on Thu Nov 28 01:04:17 2019

preprocessing template

@author: mohamedhozayen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

"""
    Demo:
        
    import numpy as np
    import pandas as pd
    import preprocessing as prc
        
    train = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
    train = train.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
    
    # clone negative class - for testing purposes
    preprocess_negative = train[train['class'] == 0]
    
    # clone positive class - for testing purposes
    preprocess_positive = train[train['class'] == 1]
    
    preprocess_negative = prc.detect_outlier_iterative_IQR(preprocess_negative)
    preprocess_negative = preprocess_negative.dropna(thresh=20) #keep rows with at least 20 non-missing values
    
    preprocess_positive = prc.detect_outlier_iterative_IQR(preprocess_positive)
    preprocess_positive = preprocess_positive.dropna(thresh=20) #keep rows with at least 20 non-missing values

"""



"""
    Iterative IQR Outlier Detection
    
    apply median +/- 1.5*IQR until no significant detection
    
    assume last columns is class type
    
    return new df
"""
def detect_outlier_iterative_IQR(og_df, cutoff_ratio = 1.07):
#    from sklearn.impute import SimpleImputer
#    imp = SimpleImputer(missing_values=-1, strategy='mean')
    df = og_df.copy()
    df_prev = pd.DataFrame({'' : []})
    n1 = 1.0
    while True:
        stat = df.describe()
        df_prev = df.copy()
        for column in df.iloc[:, :-1]:
            median, min25, max75 = stat[column]['50%'], stat[column]['25%'], stat[column]['75%']
            iqr = abs(max75-min25)*1.5
            upper_bound = median + iqr
            lower_bound = median - iqr
            df[column] = df[column].mask(df[column] > upper_bound, np.nan)
            df[column] = df[column].mask(df[column] < lower_bound, np.nan)
        n2 = df.isnull().sum().sum()
        r = n2/n1
        n1 = n2*1.0
        if r < cutoff_ratio:
            break
    return df_prev


"""
    cut threshold for extreme values by droping a the corrosponding sample
    assume last columns is class type
    return new dataframe 
"""
def cut_threshold(og_df, t):
    df = og_df.copy()
    for column in df.iloc[:, :-1]:
        i_high = df[df[column] > t].index
        i_low = df[df[column] < -t].index
        df.drop(i_high, inplace=True)
        df.drop(i_low, inplace=True)
    return df

"""
    compare medians for two dataframes
 
    return a new dataframe with features, median 1, median 2
"""
def median_stat(df_positive_stat, df_negative_stat):
    median50 = []
    for c in df_positive_stat.columns:
        median50.append([c, df_positive_stat[c]['50%'], df_negative_stat[c]['50%']])
    median50 = pd.DataFrame(median50)
    median50.columns = ['features', 'median Positive', 'median negative']
    return median50


