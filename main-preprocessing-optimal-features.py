# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:51:46 2019

@author: Mohamed Hozayen
"""

import numpy as np
import pandas as pd
import preprocessing as prc
import feature_selection as fs
import matplotlib.pyplot as plt
import seaborn as sns
import time
start_time = time.time()

df = pd.read_csv('Files\csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df))
df = prc.standarize(df) # or normalize

pca_cos_features = fs.pca_kernel(df, kernel='cosine')
features = pca_cos_features[['pca-cosine13', 'pca-cosine18', 'pca-cosine15', 'pca-cosine12', 'pca-cosine26']]

print("--- %s seconds ---" % (time.time() - start_time))

"""
Decision Tree parameters:
    Features to use : features
    max depth : 4
"""