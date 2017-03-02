# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 07:42:20 2017

@author: abrown09
"""

import pandas as pd
from scipy.cluster.vq import kmeans,vq
import numpy as np
from scipy.cluster import vq
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/amybrown/Thinkful/Unit_4/Lesson_3/curric-data-001-data-sets/un/un.csv')
# there are 207 rows in the dataset

# determine number of non-null values in each column
nonnull = df.count()

# get data types for each column
for col in df:
    print(col, df[col].dtypes)
    
final_df = df[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]

# store feature data in arrays
f1 = final_df['lifeMale'].values
f2 = final_df['lifeFemale'].values
f3 = final_df['infantMortality'].values
gdp = final_df['GDPperCapita'].values

#y = df['GDPperCapita'].values

X = np.matrix(zip(f1,f2,f3, gdp))  

X = np.vstack([f1, f2, f3, gdp])

#y = np.matrix(zip(y))
#y = np.vstack([y])
#### Clustering attempt 1 ####

km = kmeans(n_clusters=10).fit(X)

#### Clustering attempt 2 ####
np.isnan(X).any()
np.isinf(X).any()

X = np.nan_to_num(X)


centroids, variance = vq.kmeans(X, 2)

identified, distance = vq.vq(X, centroids)

cluster_1 = X[identified == 0]
cluster_2 = X[identified == 1]

print(cluster_1[0:3])
print(cluster_2[0:3])


### clustering attempt 3 ####
from sklearn.cluster import KMeans

clustering = KMeans(n_clusters=10, n_init=10, random_state=1)
clustering.fit(X)



### clustering attempt 4: start by plotting 3 features against GDP per capita ###

np.isnan(X).any()
np.isinf(X).any()

#X = np.nan_to_num(X)
#y = np.nan_to

#from sklearn import preprocessing

#std_scale = preprocessing.StandardScaler().fit(final_df)[['lifeMale', 'lifeFemale', 'infantMortality']]

#gdp=df[['GDPperCapita']]

#plt.scatter(gdp, final_df['lifeMale'])
#plt.scatter(gdp, final_df['lifeFemale'])
#plt.scatter(gdp, final_df['infantMortality'])

X = X.reshape(207, 4)

centers = []
labels = []

for k in range(1,11):
    kmeans = KMeans(n_clusters = k).fit(X)
    centers.append(kmeans.cluster_centers_) 
    labels.append(kmeans.labels_)
    


# need to put the output somewhere    
    
testk = KMeans(n_clusters = 2).fit(X)
testk.cluster_centers_
testk.labels_


### attempt 5 ###
from scipy.spatial.distance import cdist, pdist

K = range(1,11)
KM = [KMeans(n_clusters=k).fit(X) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgwithinSS = [sum(d)/X.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgwithinSS, 'b*-')
ax.plot(K[kIdx], avgwithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')


# now, cluster!
# calculate centroids of each cluster
# and the variance of all the clusters
centroids, variance = vq.kmeans(X, 3)

# seperate into clusters
identified, distance = vq.vq(X, centroids)

cluster_1 = X[identified == 0]
cluster_2 = X[identified == 1]
cluster_3 = X[identified == 2]

plt.scatter(X[:,0], X[:,1], s=100, c=identified)