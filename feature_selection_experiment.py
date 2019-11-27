from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


df = pd.read_csv("csv_result-Descriptors_Training.csv")

df_pca = pd.DataFrame()

df[:].fillna(0, inplace=True)

# sns.set()
# plt.title("Distribution of Feature 15")
# sns.distplot(df['Pb_NO_sideR35_S'])
# plt.show()

# Split into train and test
X = df.drop(['id', 'class'], axis=1)
Y = df['class']
df_pca['class'] = df['class']

print(X.shape, Y.shape)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)
df_pca['pca-one'] = pca_result[:,0]
df_pca['pca-two'] = pca_result[:,1] 
df_pca['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,7))
# ax1 = plt.subplot(1, 2, 1)
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="class",
#     palette=sns.color_palette("hls", 2),
#     data=df_pca,
#     legend="full",
#     alpha=0.6,
#     ax=ax1
# )
# ax2 = plt.subplot(1, 2, 2)
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="class",
#     palette=sns.color_palette("hls", 2),
#     data=df,
#     legend="full",
#     alpha=0.3,
#     ax=ax2
# )

sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.5
)
plt.show()
# # X_temp =  pd.DataFrame(VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(X, y))
# # X_new = pd.DataFrame(SelectKBest(f_classif, k=10).fit_transform(X_temp, y))
# X_new = pd.DataFrame(VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(X, y))
# print("Shape of original dataset: " + str(X.shape))
# print("Shape of feature selected dataset: " + str(X_new.shape))