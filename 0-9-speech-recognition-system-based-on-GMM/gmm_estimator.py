#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 核心代码，提供GMM训练和测试的代码，程序最终输出一个acc.txt文件，记录了识别准确率
import numpy as np
from utils import *
import scipy.cluster.vq as vq
from matplotlib import pyplot as plt
import time


num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
"""self代表类的实例，而非类
类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。
self 不是 python 关键字，我们把他换成 runoob 也是可以正常执行的:"""

# gmms[target] = GMM(39, K=num_gaussian)
class GMM:
    # 初始化 11个模型都进行初始化，11个mu，11个pi，11个sigma
    def __init__(self, D, K=5):    # （√）
        print("----------------------------GMM初始化参数-------------------------------------")
        start_gmm_initialization_time = time.clock()
        # print("start_gmm_initialization_time: ", start_gmm_initialization_time)
        assert (D > 0)
        self.dim = D
        self.K = K
        # Kmeans Initial
        self.mu, self.sigma, self.pi = self.kmeans_initial()

        # print("mu", self.mu)  # (5,39)
        print("mu.shape: ", np.array(self.mu).shape)

        # print("sigma", self.sigma)  #
        print("sigma.shape: ", np.array(self.sigma).shape)  # (5,39,39)

        print("pi:", self.pi)
        print("pi.shape: ", np.array(self.pi).shape)  # (1，5)
        end_gmm_initialization_time = time.clock()
        # print(" end_gmm_initialization_time: ", end_gmm_initialization_time)
        print(" gmm_initialization time: ", end_gmm_initialization_time - start_gmm_initialization_time)
        print("----------------------------GMM初始化结束-------------------------------------")

    # 使用聚类的方法获得初始化参数（√）用K-means算法输出的聚类中心，作为高斯混合模型的输入
    def kmeans_initial(self):    # （√）
        print("----------------------------K-Means初始化算法-------------------------------------")
        start_kmeans_time = time.clock()
        # print(" start_kmeans_time: ", start_kmeans_time)
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')                                   # (18593, 39)
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)  # 聚类，分成k类；centroids  质心，
        # labels [3 3 3 .0 4 .. 2 2 1]      01234   5个类比的标签 labels.shape (18593,)   通过的观测值与生成的质心之间的平均（非平方）欧氏距离
        # centroids.shape)   # (5, 39)      由k个质心组成的（k * N）数组
        clusters = [[] for i in range(self.K)]  # [[], [], [], [], []]
        for (l, d) in zip(labels, data):        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            clusters[l].append(d)               # 将标签一样的数据放在一起，5类

        for cluster in clusters:                 # 每一类中
            mu.append(np.mean(cluster, axis=0))       # 计算39维每维度的mu
            sigma.append(np.cov(cluster, rowvar=0))   # 计算两两数据协方差
        pi = np.array([len(c) * 1.0 / len(data) for c in clusters])   # 每个类数据长度/总长度=初始概率，这样做没问题
        #  [0.19797773 0.15618781 0.28957134 0.16785887 0.18840424]   初始值
        end_kmeans_time = time.clock()
        # print(" end_kmeans_time: ", end_kmeans_time)
        print(" kmeans time: ", end_kmeans_time - start_kmeans_time)
        print("----------------------------K-Means初始化算法结束-------------------------------------")
        return mu, sigma, pi  # (39,) * 5  ；(39, 39) * 5；  (5,)

    # 计算高斯概率
    def gaussian(self, x, mu, sigma):  # 一个x，总xn个  # （√）
        """Calculate gaussion probability.
    
            :param x: The observed data, dim *1. ，dim维度39；39* 345帧，共345个x,x[1]=[1,2,3,...39]
            :param mu: The mean vector of gaussian, dim*1  （39，） * 5
            :param sigma: The covariance matrix, dim*dim  （39，39）  * 5
            :return: the gaussion probability, scalor
        """
        """
         1、矩阵bai乘法，例如np.dot(X,X.T)。
         2、点积，du比如np.dot([1,2,3],[4,5,6]) = 1*4 + 2*5 + 3*6  = 32。
         一、np.linalg.det():矩阵求行列式
         二、np.linalg.inv()：矩阵求逆
         三、np.linalg.norm():求范数
         """
        #
        # print("----------------------------计算gaussian概率-------------------------------------")
        D = x.shape[0]  # D=39
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma + 0.0001)   # 防止除0错误
        mahalanobis = np.dot(np.transpose(x - mu), inv_sigma)  # np.transpose  转置
        mahalanobis = np.dot(mahalanobis, (x - mu))
        const = 1 / ((2 * np.pi) ** (D / 2))
        prob = const * (det_sigma) ** (-0.5) * np.exp(-0.5 * mahalanobis)
        # print("----------------------------计算gaussian概率结束-------------------------------------")
        return prob

    # 计算对数似然概率(√)
    def calc_log_likelihood(self, X):              # （√）
        """Calculate log likelihood of GMM

                  param: X: A matrix including data samples, num_samples * D
                  return: log likelihood of current model
              """
        # print("----------------------------计算对数似然概率-------------------------------------")
        log_llh = 0.0
        N = X.shape[0]
        for n in range(N):
            sub_n = 0.0
            for k in range(self.K):
                sub_n += self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])
            log_llh += np.log(sub_n)

        # print("----------------------------计算对数似然概率结束-------------------------------------")
        return log_llh       # 一个概率值

    # EM算法实现(√)
    def em_estimator(self, X):
        """Update paramters of GMM

                   param: X: A matrix including data samples, num_samples * D  1739* 39
                   return: log likelihood of updated model
               """
        print("----------------------------进行EM算法------------------------------------")
        log_llh = 0.0
        # E-step
        # 1、计算gama     # （√）
        N = X.shape[0]    # 1937  帧，每帧39    (1937,39)
        gama = np.zeros((self.K, N))
        for n in range(N):
            for k in range(self.K):
                gama[k, n] = self.pi[k] * self.gaussian(X[n], self.mu[k], self.sigma[k])

        gama = gama / np.sum(gama, axis=0)
        print("gama: ", gama)
        print("gama.shape: ", np.array(gama).shape)  # (5, 1937)

        # M-step
        # 2、计算NK
        Nk = np.sum(gama, axis=1)   # （√）
        # print("NK： ", Nk)             # NK [599.86507677 301.07091686 272.94131332 454.99414833 308.12854472]
        # print("NK.shape： ", np.array(Nk).shape)  # (1,5)

        # self.mu = np.dot(gama,X)
        for k in range(self.K):
            self.sigma[k] = np.zeros((self.dim, self.dim))   # 39*39
            # 3、计算mu
            self.mu[k] = np.dot(gama[k], X) / Nk[k]     # （1,1937）* (1937,39)=（1,39）*5 = (5,39)
            for n in range(N):
                # 4、计算sigma  np.outer 对于多维向量，全部展开变为一维向量,用来求外积的
                """
                a = [a1, …, am] and b = [ b1, …, bn]
                result=[[a1*b1, a1*b2,…,a1*bn]
                [a2*b1, a2*b2,…,a2*bn]…
                [am*b1, am*b2,…,am*bn]]
                """
                self.sigma[k] += gama[k, n] * np.outer((X[n] - self.mu[k]), X[n] - self.mu[k])
            self.sigma[k] /= Nk[k]            # (1,39,39)*5 = (5,39,39)
        # print(self.sigma )
        # print(self.sigma[0].shape)
        # 5、计算pi
        self.pi = Nk / N

        log_llh = self.calc_log_likelihood(X)   # 参数已经更新
        print(" log_llh: ", log_llh)   #  log_llh  -175252.53056248167->-174154.29695584765...五次，到下一个标签，继续5次，共5*11=55次

        print(" self.pi: ", self.pi)   # [0.1464419  0.25219225 0.11557325 0.28323082 0.20256178]
        print("self.pi.shape: ", np.array(self.pi).shape)   # (5,)

        # print("  self.sigma: ", self.sigma)
        print("self.sigma.shape: ", np.array(self.sigma).shape)   #  (5, 39, 39)

        # print("  self.mu ", self.mu)
        print("self.mu.shape: ", np.array(self.mu).shape)    #  (5, 39)

        print("NK: ", Nk)   # NK [283.65796789 488.49637872 223.86537666 548.61810247 392.36217426]
        print("NK.shape: ", np.array(Nk).shape)   # (5,)
        print("----------------------------EM算法结束-------------------------------------")
        return log_llh


def train(gmms, num_iterations=num_iterations):   # 5次迭代
    print("-------------------------------------------------开始训练------------------------------------------------------")
    start_train_time = time.clock()
    # print(" start_train_time: ",  start_train_time)
    dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    print(" dict_utt2feat: ",  dict_utt2feat)    # 语音  对应 的特征{'KK1B_endpt.wav': 'train/feats.ark:2244453', 'GR5A_endpt.wav': 'train/feats.ark:1399365',
    print(" dict_target2utt: ", dict_target2utt)  # 标签 对应 的语音 {'O': ['AEOA_endpt.wav', 'AGOA_endpt.wav', 'AWOA_endpt.wav', 'BDOA_endpt.wav', 'BDOB_endpt.wav', 'BROB
    # targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    for target in targets:   # 11次循环
        feats = get_feats(target, dict_utt2feat, dict_target2utt)  # 获得属于一个类（target）的特征数据feats（samples_num，39）
        # print(" feats", feats)
        print(" feats.shape： ", feats.shape)  # (1937, 39)  一次循环，一个target
        for i in range(num_iterations):  # 训练迭代次数5次
            print("---------------------------------开始迭代---------------------------------------")
            print(" target： ", target)
            print(" 迭代次数", i+1)
            log_llh = gmms[target].em_estimator(feats)  # 用属于对应target的GMM的EM算法训练参数 1,2,3.。。都要训练
            print(" log_llh_train： ", log_llh)  # -175242.91 ->-174176.0->-174176.0->-173818.1->-173616.910  (共五次迭代)
    end_train_time = time.clock()
    # print(" end_train_time: ", end_train_time)
    print(" training time: ", end_train_time - start_train_time)
    print("--------------------------------------------------训练结束--------------------------------------------------------")

    return gmms


def a_test_a(gmms):
    print("------------------------------------------------------开始测试--------------------------------------------------------")
    start_test_time = time.clock()
    print("start_test_time: ", start_test_time)
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    print(" dict_utt2feat_test： ", dict_utt2feat)       # 语音 对应 特征 {'JE8B_endpt.wav': 'test/feats.ark:597759',
    print(" dict_target2utt_test： ", dict_target2utt)   # 标签 对应 语音 {'O': ['AEOA_endpt.wav', 'AGOA_endpt.wav'
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]    # utt  ACZB_endpt.wav   读数字key,实际上是读取key对应的东西
        for utt in utts:
            dict_utt2target[utt] = target   # 又重新建立一个字典，key是uut语音名，对应的是target数字
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])  # kaidi读取特征  训练时也是
        # print(" feats_test", feats)
        print(" feats_test.shape： ", feats.shape)   # (42, 39) (64, 39) (84, 39)(46, 39)
        scores = []
        for target in targets:     # 0123456789o
            scores.append(gmms[target].calc_log_likelihood(feats))  # 预测 1739*39
            print("模型", target)
            print("calc_log_likelihood： ", gmms[target].calc_log_likelihood(feats))
        print("scores： ", scores)
        predict_target = targets[scores.index(max(scores))]    # 每个模型的得分最大，就判断为哪个
        print("predict_target： ", predict_target)
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
        acc_pass = correction_num * 1.0 / (correction_num + error_num)
        print("acc_pass： ", acc_pass)
    acc = correction_num * 1.0 / (correction_num + error_num)
    print("acc： ", acc)
    end_test_time = time.clock()
    print(" end_test_time: ", end_test_time)
    print(" test time: ", end_test_time - start_test_time)
    print("------------------------------------------------------测试结束-------------------------------------------------------------")

    return acc



def main():
    print("---------------------------------进入主函数Main()--------------------------------------")
    start_system_time = time.clock()
    print(" start_system_time: ", start_system_time)
    gmms = {}   # 字典
    # targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    for target in targets:   # 哦。。。在这里循环初始化11个模型
        print("target： ", target)
        gmms[target] = GMM(39, K=num_gaussian)  # Initial model dict{str:gmm} 每个数字都有自己的GMM模型，类的实例化  Initial model   为每一个类型数据创建一个GMM模型，保存在gmms(字典)

    print("gmms{}： ", gmms)     # {'2': <__main__.GMM object at 0x0000026B354A5F98>, '7': <__main__.GMM object at 0x0000026B354A5400>, 'Z': <__main_
    gmms = train(gmms)  # 在训练集上训练gmms中的混合高斯模型
    acc = a_test_a(gmms)  # 在测试集上测试模型
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()
    end_system_time = time.clock()
    print(" end_system_time: ", end_system_time)
    print(" system time: ", end_system_time - start_system_time)
    print("---------------------------------主函数Main()结束--------------------------------------")

if __name__ == '__main__':
    print("---------------------------------进入name--------------------------------------")
    main()
    print("---------------------------------name结束--------------------------------------")



