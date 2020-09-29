#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class GMM:
    def __init__(self, D, K=5):   # （√）
        assert (D > 0)
        self.dim = D
        self.K = K
        # Kmeans Initial
        self.mu, self.sigma, self.pi = self.kmeans_initial()
        print("---------------------------------- mu -------------------------------")
        print("mu", self.mu)  # (5,39)
        print("mu.shape", np.array(self.mu).shape)
        print("---------------------------------- sigma ----------------------------")
        print("sigma", self.sigma)    #
        print("sigma.shape", np.array(self.sigma).shape)   # (5,39,39)
        print("------------------------------------ pi ------------------------------")
        print("pi", self.pi)
        print("pi.shape", np.array(self.pi).shape)   # (5,)

    # 用K-means算法输出的聚类中心，作为高斯混合模型的输入
    def kmeans_initial(self):     # （√）
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')                                   # (18593, 39)
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)  # 聚类，分成k类
        # labels [3 3 3 0 4 .. 2 2 1]  即01234 这5个类别的标签 ; labels.shape (18593,)
        # centroids  由k个质心组成的（k * N）数组 ; centroids.shape (5, 39)
        clusters = [[] for i in range(self.K)]                 # [[], [], [], [], []]
        for (l, d) in zip(labels, data):
            clusters[l].append(d)                              # 将标签一样的数据放在一起，5类

        for cluster in clusters:                               # 在每一类中计算39维每维度的mu，协方差sigma
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c) * 1.0 / len(data) for c in clusters])  # 每个类数据长度/总长度=初始概率
        # pi [0.19797773 0.15618781 0.28957134 0.16785887 0.18840424]
        return mu, sigma, pi                                    # (1,39) * 5 ;(39, 39) * 5 ;  (1,5)

    # 计算高斯概率
    def gaussian(self, x, mu, sigma):      # 一个x，总xn个  # （√）
        """Calculate gaussion probability.
        :param x: The observed data, dim *1.   dim维度39；39* 帧，共n个x,x[1]=[1,2,3,...39]
        :param mu: The mean vector of gaussian, dim*1  （39，） * 5
        :param sigma: The covariance matrix, dim*dim  （39，39）  * 5
        :return: the gaussion probability, scalor
        """
        D = x.shape[0]
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)
        mahalanobis = np.dot(np.transpose(x - mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x - mu))
        const = 1 / ((2 * np.pi) ** (D / 2))
        return const * (det_sigma) ** (-0.5) * np.exp(-0.5 * mahalanobis)

    # 计算对数似然概率(√)
    def calc_log_likelihood(self, X):      # （√）
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model
        """
        log_llh = 0.0
        N = X.shape[0]
        for n in range(N):
            sub_n = 0.0
            for k in range(self.K):
                sub_n += self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
            log_llh += np.log(sub_n)
        return log_llh                   # 一个概率值


    # EM算法实现(√)
    def em_estimator(self, X):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model 
        """
        log_llh = 0.0
        # 一、E-step
        # 1、计算后验概率gama
        N = X.shape[0]                  # 1937帧，每帧39维
        gama = np.zeros((self.K, N))
        for n in range(N):
            for k in range(self.K):
                gama[k, n] = self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
                # print("gama", gama)
        gama = gama / np.sum(gama, axis=0)    # (5, 1937)

        # 二、M-step
        # 2、计算NK
        Nk = np.sum(gama, axis=1)  # NK [599.8  301.0  272.9  454.9  308.1]    (1,5)
        for k in range(self.K):
            self.sigma[k] = np.zeros((self.dim, self.dim))    # （39，39）
            # 3、计算mu
            self.mu[k] = np.dot(gama[k], X) / Nk[k]           # （1,1937）* (1937,39)=（1,39）*5 = (5,39)
            for n in range(N):
                # 4、计算sigma
                self.sigma[k] += gama[k, n] * np.outer((X[n] - self.mu[k]), X[n] - self.mu[k])
            self.sigma[k] /= Nk[k]                            # (1,39,39)*5 = (5,39,39)
        # 5、计算pi
        self.pi = Nk / N            # # [0.1464419  0.25219225 0.11557325 0.28323082 0.20256178]
        log_llh = self.calc_log_likelihood(X)
        # -175242.91 -> -174176.0 -> -174176.0 -> -173818.1 -> -173616.910  (共五次迭代)
        return log_llh


def train(gmms, num_iterations=num_iterations):
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')

    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)  #
        # print(feats)
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
            print(i)
            print(log_llh)
    return gmms


def a_test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian)  # Initial model
    gmms = train(gmms)
    acc = a_test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()
