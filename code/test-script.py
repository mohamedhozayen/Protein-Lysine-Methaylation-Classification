"""
Steps:
    Detect outliers
    handle outliers
    Normalization/standarization
    
    Unsupervised techniques
        variance threshold
        PCA 
        PCA-Kernal
        
    Balance data
    
    Supervised techniques
        Filter 
        Wrapper
            sklearn.feature_selection.RFE
            sklearn.feature_selection.RFECV
    
    Meta learning
    Test using imbalance set

"""
from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import preprocessing as prc
import feature_selection as fs


df = pd.read_csv('Files\csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.handle_outlier(prc.detect_outlier_iterative_IQR(df).dropna(thresh=20))
df = prc.standarize(df) #normalize

pca = fs.pca_linear(df, n_c=2, pos_only=False) 

# return all features with at least thershold 
# no selection below 1 !!!
fs_vairance = fs.variance_threshold(df, threshold=1)
fs_vairance = pd.concat([fs_vairance, df['class']], axis=1)


pca, df_pca = fs.pca_linear(df, n_c=9, pos_only=False)
df_pca.to_csv('fs-pca-9.csv')

pca_pca, df_pca_pca = fs.pca_linear(df_pca, n_c=2, pos_only=True, plot=True) 


"""
    do kernel pca 
    run with all data -> ask ben, gpu
    run with fs_variance
    run with pca-9
"""



# =============================================================================
# DONT LOOK BEYOND HERE :)
# =============================================================================
"""

pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()


fs_vairance.to_csv('fs-variance-threshold.csv')


features = df_norm.iloc[:,:-1]
target= df_norm.iloc[:,-1]

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
f_anova = fs.select_k_best(features, target, f_classif, 7)
df = df_norm[f_anova.iloc[:,0].append(pd.Series('class'))] #

f_anova.to_csv('f_anova-scores.csv')
pearson.to_csv('pearson-ranking.csv')
spearman.to_csv('spearman-ranking.csv')

Pb_NO_sideR35_S	14
Z3_NO_UCR_S	    19
Z3_NO_UCR_N1	20
Z3_NO_NPR_V	    24
IP_NO_PLR_S     25

for column in df.iloc[:, :-1].columns:
    if column not in d:
        p_stat.drop(column, inplace=True, axis=1)
        n_stat.drop(column, inplace=True, axis=1)
        
Figures = 'Figures/'
Files = 'Files/'
val = pd.read_csv('csv_result-Descriptors_Calibration.csv', sep=',')

train = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
train = train.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])

#baseline samples - no changing
#seperate positive and negative class
t_feature_positive = train[train['class'] == 1]
t_feature_positive = t_feature_positive.drop(['class'], axis=1)
t_feature_negative = train[train['class'] == 0]
t_feature_negative = t_feature_negative.drop(['class'], axis=1)

# clone negative class - for testing purposes
preprocess_negative = train[train['class'] == 0]
preprocess_negative = preprocess_negative.drop(['class'], axis=1)

# clone positive class - for testing purposes
preprocess_positive = train[train['class'] == 1]
preprocess_positive = preprocess_positive.drop(['class'], axis=1)


m_b = prc.median_stat(t_feature_positive.describe(), preprocess_negative.describe())


#preprocess_negative = prc.cut_threshold(preprocess_negative, 500)
preprocess_negative = prc.detect_outlier_iterative_IQR(preprocess_negative)
preprocess_negative = preprocess_negative.dropna(thresh=20) #keep rows with at least 20 non-missing values

preprocess_positive = prc.detect_outlier_iterative_IQR(preprocess_positive)
preprocess_positive = preprocess_positive.dropna(thresh=20) #keep rows with at least 20 non-missing values

m_a = prc.median_stat(preprocess_positive.describe(), preprocess_negative.describe())


df = pd.read_csv('csv_result-Descriptors_Training.csv', sep=',') 
df = df.drop(['id'], axis=1).replace(['P', 'N'], [1, 0])
df = prc.detect_outlier_iterative_IQR(df)
df = df.dropna(thresh=20).fillna(0)  

(preprocess_negative == -1).sum() count -1
preprocess_negative.isnull().sum().sum() # nan count
n_stat = preprocess_negative.describe()


df.to_csv('.csv')

 Z2_NO_AHR_CV


for index, row in preprocess_negative.iterrows():
        print(preprocess_negative.iloc[index, :-1] )
        
IQR argument
https://tolstoy.newcastle.edu.au/R/help/05/07/8210.html
>>
>> --- 95% (2 std)
>> |
>> |
>> ------- 75%
>> | |
>> |-----| 50%
>> | |
>> | |
>> ------- 25%
>> |
>> --- 5% (2 std)

1. inspect negative class stat, check median differences
2. cut extrem values outside cuttoff threshold t
3. inspect negative class stat, check median differences
4. handle leftover outliers 



preprocess_negative.describe().to_csv('neg-stat-after.csv')
preprocess_negative.describe().to_csv('neg-stat-before.csv')


from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

fill_NaN = SimpleImputer(missing_values=-1, strategy='mean')
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(preprocess_negative))
imputed_DF.columns = preprocess_negative.columns
imputed_DF.index = preprocess_negative.index


def drop(df, cuttoff):
    for column in df.columns:
        df[column] = df[column].apply(lambda x: [y if y <= 9 else 11 for y in x])

for column in t_feature_negative.columns:
    check = preprocess_negative[column].between(-100, 100)


    [x for x in arr if (x > mean - 2 * sd)]
    
        and df[column] < lower_bound
        df[column] = df[column].apply(lambda x: [y if y < upper_bound else -1 for y in x])
        df[column] = df[column].apply(lambda x: [y if y > lower_bound else -1 for y in x])    
    final_list = [x for x in arr if (x > mean - 2 * sd)]

        preprocess_negative = preprocess_negative[preprocess_negative[column] < mean + max75]
        preprocess_negative = preprocess_negative[preprocess_negative[column] > mean - min25]
    preprocess_negative = preprocess_negative[preprocess_negative[column] < (median + iqr)]
    preprocess_negative = preprocess_negative[preprocess_negative[column] > (median - iqr)]
    preprocess_negative = preprocess_negative[preprocess_negative[column] < (median + 3.0*std)]
    preprocess_negative = preprocess_negative[preprocess_negative[column] > (median - 3.0*std)]
           median = stat[column]['50%']
        min25 = stat[column]['25%']
        max75 = stat[column]['75%']
        mean = stat[column]['mean']
        std = stat[column]['std']
        iqr = (max75-min25)*1.5
        max_val = stat[column]['max']
        min_val = stat[column]['min'] 
(preprocess_negative == -1).sum() count -1

    
t_feature.iloc[:,0]
len([i for i in t_feature[t_feature.columns[1]] if i > 100])

from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
  eps = 0.5,
  metric="euclidean",
  min_samples = 20,
  n_jobs = -1)
clusters = outlier_detection.fit_predict(l)

from matplotlib import cm
cmap = cm.get_cmap('Accent')
l.plot.scatter(
  x = t_feature.columns[1],
  y = "class",
  c = clusters,
  cmap = cmap
)


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



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
            
        

"""