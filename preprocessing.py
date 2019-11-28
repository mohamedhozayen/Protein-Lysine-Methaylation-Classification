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
    under development...
    
    replace outliers with -1. 
    outliers outside mean +/- 1.5*iqr
    use imputation to replace -1
 
    return new df
"""
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

    return df

"""
    cut threshold for extreme values by droping a the corrosponding sample
    
    return new dataframe 
"""
def cut_threshold(og_df, t):
    df = og_df.copy()
    for column in df.columns:
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


