# 使用聚类的方法获得初始化参数（√）用K-means算法输出的聚类中心，作为高斯混合模型的输入
def kmeans_initial(self):  # （√）
    print("----------------------------K-Means初始化算法-------------------------------------")
    mu = []
    sigma = []
    data = read_all_data('train/feats.scp')  # (18593, 39)
    (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)  # 聚类，分成k类；centroids  质心，
    # labels [3 3 3 .0 4 .. 2 2 1]      01234   5个类比的标签 labels.shape (18593,)   通过的观测值与生成的质心之间的平均（非平方）欧氏距离
    # centroids.shape)   # (5, 39)      由k个质心组成的（k * N）数组
    clusters = [[] for i in range(self.K)]  # [[], [], [], [], []]
    for (l, d) in zip(labels, data):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        clusters[l].append(d)  # 将标签一样的数据放在一起，5类

    for cluster in clusters:  # 每一类中
        mu.append(np.mean(cluster, axis=0))  # 计算39维每维度的mu
        sigma.append(np.cov(cluster, rowvar=0))  # 计算两两数据协方差
    pi = np.array([len(c) * 1.0 / len(data) for c in clusters])  # 每个类数据长度/总长度=初始概率，这样做没问题
    #  [0.19797773 0.15618781 0.28957134 0.16785887 0.18840424]   初始值
    print("----------------------------K-Means初始化算法结束-------------------------------------")
    return mu, sigma, pi  # (39,) * 5  ；(39, 39) * 5；  (5,)
