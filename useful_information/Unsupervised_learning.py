'''
非监督学习
Unsupervised_learning.py
'''


# 1 <聚类>clustering
from sklearn.cluster import *
# 1.1 <k-means>
'''
伪代码：
1. 随机选择K个样本作为初始均值向量
2. 重复直到均值向量未更新：
3.    计算各个样本到各个均值向量的距离，选择最近的作为类标记
4.    计算新的均值向量，若有变化就更新


'''
KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)
'''
n_clusters：这个参数指出聚类簇的个数即k的值；
init：默认k-means++，可以选择较分散的点作为初始聚类中心点，还可以选择random，表示随机选择初始点作为聚类中心迭代。
'''

MiniBatchKMeans(n_clusters=8, init=’k-means++’, max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
# mini-batch思想：每次迭代从样本中随机选择一部分的样本来训练。其效果只比不mini-batch的差一点，但训练速度将加快很多。

# 1.2 <学习向量量化>（Learning Vector Quantization,LVQ)
# 1.3 <高斯混合聚类>(Mixture-of-Gaussian)
# 2 <密度估计>density estimation

# 3 <异常检测>anomaly detection
