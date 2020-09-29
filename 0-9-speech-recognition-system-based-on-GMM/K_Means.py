#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from utils import *
import scipy.cluster.vq as vq
# from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt

K = 5
mu = []
sigma = []
data = read_all_data('train/feats.scp')
# print("data", data)
print("data.shape", data.shape)  # (18593, 39)  这么多的39维特征

# 对形成k个群集的一组观察向量执行k均值。 这样便产生了一个将质心映射到代码的codebook
(centroids, labels) = vq.kmeans2(data, K, minit="points", iter=100)  # 聚类，分成k类；此参数不代表k均值算法的迭代次数 centroids  质心，

# data = whiten(data)
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
plt.show()

# print("centroids", centroids)
print("centroids.shape", centroids.shape)  # (5, 39) 由k个质心组成的（k * N）数组
print("labels[100]", labels[103])  # labels [3 3 3 ... 2 2 1]    是01234，没显示全
print("labels.shape", labels.shape)  # labels.shape (18593,)

clusters = [[] for i in range(K)]  # 5个集群
print("clusters", clusters)  # clusters [[], [], [], [], []]

print("(labels, data)", (labels, data))
print("zip(labels, data)", zip(labels, data))

# labels与data一一对应，labels.shape (18593,)，data.shape (18593, 39) ;标签打的是什么？
for (l, d) in zip(labels, data):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    clusters[l].append(d)   # （18593,39）,18593个标签，每个标签里面39个数字,把标签一样的放在一起
    # clusters[l] [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]
    # print("l", l)   # 3
    # print("d", d)   # (39,)
    # print("d.shape", d.shape)  # d,shape (39,)
    # print("clusters[l]", clusters[l])
    # print("clusters[l].shape", np.array(clusters[l]).shape)    # (4060, 39) (3023, 39) 。。5堆 不等，4060个相同的标签l,对应39维特征
# print("clusters[]", clusters)
for cluster in clusters:      # clusters   3维   5个
    # print("cluster", cluster)
    mu.append(np.mean(cluster, axis=0))   # 一堆，39维，各维度的mu
    print("mu", mu)
    print("mu.shape", np.array(mu).shape)   # (1,39)* 5 =（5,39）
    sigma.append(np.cov(cluster, rowvar=0))
    print("sigma", sigma)
    print("sigma.shape", np.array(sigma).shape)   # (1,39,39)*5 =（5,39,39）  两两数据之间的相关性
pi = np.array([len(c) * 1.0 / len(data) for c in clusters])
print("mu[0]", mu[0])
print("mu[0].shape", mu[0].shape)  # (39,)  * 5
print("sigma[0]", sigma[0])
print("sigma[0].shape", sigma[0].shape)  # (39, 39)   * 5
print("pi", pi)  # pi [0.19797773 0.15618781 0.28957134 0.16785887 0.18840424]
print("pi.shape", pi.shape)  # pi.shape (1，5)

"""
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
"""

# vq.kmeans2
"""
参数
obs
M×N阵列的每一行都是观察向量。 列是每次观察期间看到的特征。 必须先使用whiten将特征增白。
k_or_guess
生成的质心数。 将代码分配给每个质心，这也是质心在生成的code_book矩阵中的行索引。
通过从观察矩阵中随机选择观察值来选择初始k重心。 可替代地，将k乘以N数组指定初始的k个质心。
iter
运行k均值的次数，返回具有最低失真的代码本。 如果为k_or_guess参数的数组指定了初始质心，则将忽略此参数。 此参数不代表k均值算法的迭代次数。
thresh
如果自上次k均值迭代以来失真的变化小于或等于阈值，则终止k均值算法。
check_finite
是否检查输入矩阵仅包含有限数。 禁用可能会提高性能，但是如果输入中确实包含无穷大或NaN，则可能会导致问题（崩溃，终止）。 默认值：True
返回值
codebook
由k个质心组成的k x N数组。 第i个质心代码簿[i]用代码i表示。 生成的质心和代码表示所看到的最低失真，而未必是全局最小失真。
distortion
通过的观测值与生成的质心之间的平均（非平方）欧氏距离。 请注意，在k均值算法的上下文中，失真的标准定义有所不同，即平方距离的总和。
"""
