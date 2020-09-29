#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from utils import *
import scipy.cluster.vq as vq
clusters = [[] for i in range(5)]  # 5个集群[[], [], [], [], []]
labels = [3, 3, 3, 4, 2, 2, 2, 1, 4]
data = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8]]
for (l, d) in zip(labels, data):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    clusters[l].append(d)  # （18593,39）,18593个标签，每个标签里面39个数字
    print("l", l)  # 3
    print("d", d)  # (39,)
    print("d.shape", np.array(d).shape)  # d,shape (39,)
    print("clusters[l]", clusters[l])
print("clusters[]", clusters)
