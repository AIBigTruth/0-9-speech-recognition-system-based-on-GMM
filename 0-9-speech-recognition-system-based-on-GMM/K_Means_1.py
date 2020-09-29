#!/usr/bin/python
# -*- coding: utf-8 -*-
# Scipy聚类效果测试
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

pts = 50
a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = np.random.multivariate_normal([30, 10], [[10, 2], [2, 1]], size=pts)
# np.random.multivariate_normal这个官方解释说从多元正态分布中抽取随机样本
features = np.concatenate((a, b))
print(features)
print(features.shape)
whitened = whiten(features)
print(whitened)
codebook, distortion = kmeans(whitened, 2)  # 这个Kmeans好像只返回聚类中心、观测值和聚类中心之间的失真
print("codebook:", codebook)
print("distortion: ", distortion)
plt.scatter(whitened[:, 0], whitened[:, 1], c='g')
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()


# import numpy as np
# from scipy.cluster.vq import vq, kmeans, whiten
# import matplotlib.pyplot as plt
#
# features = np.array([[1.9, 2.0],
#                      [1.7, 2.5],
#                      [1.6, 3.1],
#                      [0.1, 0.1],
#                      [0.8, 0.3],
#                      [0.4, 0.3],
#                      [0.22, 0.1],
#                      [0.4, 0.3],
#                      [0.4, 0.5],
#                      [1.8, 1.9]])
#
# wf = whiten(features)  # 主要作用是去除数据中的冗余信息,它是obs中每一个元素除以自己所在行的标准差后得来
# print("whiten features: \n", wf)
#
# book = np.array((wf[0], wf[1]))
#
# codebook, distortion = kmeans(wf, book)
# # 可以写kmeans(wf,2)， 2表示两个质心，同时启用iter参数
# print("codebook:", codebook)
# print("distortion: ", distortion)
#
# plt.scatter(wf[:, 0], wf[:, 1])
# plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
# plt.show()
