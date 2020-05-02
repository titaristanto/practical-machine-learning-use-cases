from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc

np.random.seed(123)

A = imread('mandrill-large.tiff')
B = imread('mandrill-small.tiff')
plt.show()

k=16
rowMatA=A.shape[0]
colMatA=A.shape[1]

rowMatB=B.shape[0]
colMatB=B.shape[1]
widMatB=B.shape[2]

# Initialize k-centroids
mu=np.zeros((k, widMatB))
for i in range(k):
    r=int(np.random.rand() * rowMatB)
    c=int(np.random.rand() * colMatB)
    mu[i]=B[r,c,:]

# k-means iteration
max_iter=70
for i in range(max_iter): # loop through iterations
    pixels_in_cluster=np.zeros(k)
    miu=np.zeros((k, widMatB))
    for j in range(rowMatB): # loop through pixel rows
        for l in range(colMatB): # loop through pixel columns
            dist=np.zeros(k)
            pixel=np.reshape(B[j,l,:],(1,3))
            for m in range(k): # loop through clusters
                # calculating distance
                dist[m]=np.linalg.norm(mu[m]-pixel,2)**2
            closest_dist=min(dist)
            cluster=np.argsort(dist)[0]
            pixels_in_cluster[cluster]+=1
            miu[cluster]+=np.reshape(pixel,(3,))
    miu=[miu[m]/pixels_in_cluster[m] if pixels_in_cluster[m]>=0 else mu[m] for m in range(k)]
    mu=miu

# Applying the clustering to "mandrill-large"
A_compressed=A
for i in range(rowMatA):
    for j in range(colMatA):
        dist=np.zeros(k)
        pixelA=np.reshape(A[i,j,:],(1,3))
        for m in range(k):
            dist[m]=np.linalg.norm(mu[m]-pixelA,2)**2
        closest_dist=min(dist)
        cluster=np.argsort(dist)[0]
        A_compressed[i,j,:]=mu[cluster]
scipy.misc.imsave('outfile.jpg', A_compressed)
