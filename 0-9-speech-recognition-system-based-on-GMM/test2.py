import kaldi_io
import numpy as np


feat_scp = 'train/feats.scp'
text_file = 'train/text'
feat_fid = open(feat_scp, 'r')
text_fid = open(text_file, 'r')

feat = feat_fid.readlines()
text = text_fid.readlines()
print("feat", feat)  # ['ACZB_endpt.wav train/feats.ark:15\n', 'AE1A_endpt.wav train/feats.ark:7377\n',
print("text ", text)  # ['ACZB_endpt.wav Z\n', 'AE1A_endpt.wav 1\n', 'AE1B_endpt.wav 1\n',
feat_fid.close()
text_fid.close()
assert (len(feat) == len(text))
print("len(feat)", len(feat))    # 330  训练数据 330句话，每个字符30句话，11个字符
print("len(text)", len(text))    # 330  测试数据：110句话，每个字符10句话，11个字符
# Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
# 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。
# 等价于：
# if not true:
#   raise AssertionError
dict_utt2feat = {}   # 代表dict字典数据类型
dict_target2utt = {}
for i in range(len(feat)):
    utt_id1, ark = feat[i].strip('\n').split(' ')
    utt_id2, target = text[i].strip('\n').split(' ')
    dict_utt2feat[utt_id1] = ark     # 变成字典   {'RN5A_endpt.wav': 'train/feats.ark:2828253',
    if target in dict_target2utt.keys():
        dict_target2utt[target].append(utt_id2)   # 标签一样的放在一起
    else:
        dict_target2utt[target] = [utt_id2]
print("utt_id1", utt_id1)    # SP7A_endpt.wav
print("ark", ark)            # train/feats.ark:2902593
print("utt_id2", utt_id2)    # SP7A_endpt.wav
print("target", target)      # 7
print("dict_utt2feat", dict_utt2feat)  # dict_utt2feat {'CLOB_endpt.wav': 'train/feats.ark:517647', 'SP1B_endpt.wav'
print("dict_target2utt", dict_target2utt)   # {'1': ['AE1A_endpt.wav', 'AE1B_endpt.wav', 'AJ1A_endpt.wav', 'AN1A_endpt.wav', 'CM1B_endpt.wav', 'DC1A_endpt.wav', 'DL1B_endpt.wav', 'EH1A_endpt.wav', 'EL1B_endpt.wav', 'ES1B_endpt.wav', 'FC1B_endpt.wav', 'FL1B_endpt.wav', 'GR1B_endpt.wav', 'HN1A_endpt.wav', 'IN1A_endpt.wav', 'JC1B_endpt.wav', 'JN1B_endpt.wav', 'JP1A_endpt.wav', 'JR1A_endpt.wav', 'JT1A_endpt.wav', 'KD1A_endpt.wav', 'KK1B_endpt.wav', 'KP1B_endpt.wav', 'MM1A_endpt.wav', 'MM1B_endpt.wav', 'PK1B_endpt.wav', 'PM1A_endpt.wav', 'RA1A_endpt.wav', 'RR1B_endpt.wav', 'SP1B_endpt.wav'], '3': ['AI3A_endpt.wav', 'AL3A_endpt.wav', 'BR3A_endpt.wav', 'CL3B_endpt.wav', 'CR3A_endpt.wav', 'DL3B_endpt.wav', 'DN3A_endpt.wav', 'EL3B_endpt.wav', 'FF3A_endpt.wav', 'FK3B_endpt.wav', 'GJ3A_endpt.wav', 'GT3A_endpt.wav', 'GT3B_endpt.wav', 'HH3B_endpt.wav', 'HL3A_endpt.wav', 'HL3B_endpt.wav', 'HS3B_endpt.wav', 'IF3B_endpt.wav', 'JI3A_endpt.wav', 'JJ3B_endpt.wav', 'KT3B_endpt.wav', 'LA3A_endpt.wav', 'LD3A_endpt.wav', 'MK3B_endpt.wav', 'MS3B_endpt.wav', 'MW3B_endpt.wav', 'PM3B_endpt.wav', 'RA3A_endpt.wav', 'RA3B_endpt.wav', 'RS3A_endpt.wav'], '8': ['AI8A_endpt.wav', 'AJ8B_endpt.wav', 'CL8B_endpt.wav', 'CM8B_endpt.wav', 'CR8A_endpt.wav', 'DN8A_endpt.wav', 'DN8B_endpt.wav', 'EA8B_endpt.wav', 'EH8B_endpt.wav', 'FC8A_endpt.wav', 'FD8B_endpt.wav', 'FJ8A_endpt.wav', 'FJ8B_endpt.wav', 'GR8A_endpt.wav', 'HP8A_endpt.wav', 'HS8B_endpt.wav', 'IE8A_endpt.wav', 'IL8A_endpt.wav', 'JI8A_endpt.wav', 'JJ8A_endpt.wav', 'JR8B_endpt.wav', 'KF8B_endpt.wav', 'KH8A_endpt.wav', 'KP8B_endpt.wav', 'KR8B_endpt.wav', 'MK8B_endpt.wav', 'MM8A_endpt.wav', 'PE8A_endpt.wav', 'PP8B_endpt.wav', 'RA8B_endpt.wav'], '6': ['AN6B_endpt.wav', 'AW6A_endpt.wav', 'BR6B_endpt.wav', 'CB6A_endpt.wav', 'CF6A_endpt.wav', 'CF6B_endpt.wav', 'EA6A_endpt.wav', 'EC6A_endpt.wav', 'EH6A_endpt.wav', 'EH6B_endpt.wav', 'EK6A_endpt.wav', 'FK6A_endpt.wav', 'GG6A_endpt.wav', 'GJ6A_endpt.wav', 'GT6A_endpt.wav', 'HS6A_endpt.wav', 'IE6B_endpt.wav', 'IG6B_endpt.wav', 'IL6B_endpt.wav', 'IN6B_endpt.wav', 'IT6B_endpt.wav', 'JR6A_endpt.wav', 'KC6A_endpt.wav', 'KK6B_endpt.wav', 'KN6A_endpt.wav', 'KN6B_endpt.wav', 'LS6A_endpt.wav', 'MS6A_endpt.wav', 'NC6B_endpt.wav', 'RN6A_endpt.wav'], 'O': ['AEOA_endpt.wav', 'AGOA_endpt.wav', 'AWOA_endpt.wav', 'BDOA_endpt.wav', 'BDOB_endpt.wav', 'BROB_endpt.wav', 'CAOB_endpt.wav', 'CGOA_endpt.wav', 'CLOB_endpt.wav', 'DCOB_endpt.wav', 'DLOB_endpt.wav', 'EAOA_endpt.wav', 'EHOA_endpt.wav', 'FIOB_endpt.wav', 'GJOA_endpt.wav', 'GTOB_endpt.wav', 'HAOB_endpt.wav', 'HLOB_endpt.wav', 'HSOA_endpt.wav', 'IGOA_endpt.wav', 'JJOB_endpt.wav', 'JPOA_endpt.wav', 'JTOB_endpt.wav', 'KFOB_endpt.wav', 'KHOA_endpt.wav', 'KHOB_endpt.wav', 'KKOA_endpt.wav', 'KPOA_endpt.wav', 'KROB_endpt.wav', 'MMOB_endpt.wav'], '9': ['AI9B_endpt.wav', 'AJ9A_endpt.wav', 'AJ9B_endpt.wav', 'BR9A_endpt.wav', 'DL9B_endpt.wav', 'EC9B_endpt.wav', 'EI9A_endpt.wav', 'EK9A_endpt.wav', 'EK9B_endpt.wav', 'FC9A_endpt.wav', 'FJ9A_endpt.wav', 'FJ9B_endpt.wav', 'FK9A_endpt.wav', 'GG9A_endpt.wav', 'HH9B_endpt.wav', 'HS9A_endpt.wav', 'IE9A_endpt.wav', 'IE9B_endpt.wav', 'IF9A_endpt.wav', 'IG9B_endpt.wav', 'IH9A_endpt.wav', 'IL9A_endpt.wav', 'JN9A_endpt.wav', 'JR9A_endpt.wav', 'KF9A_endpt.wav', 'KP9A_endpt.wav', 'LA9B_endpt.wav', 'LD9B_endpt.wav', 'PP9B_endpt.wav', 'RE9B_endpt.wav'], '4': ['AG4B_endpt.wav', 'AJ4A_endpt.wav', 'AW4B_endpt.wav', 'BI4A_endpt.wav', 'CB4A_endpt.wav', 'CL4B_endpt.wav', 'CM4B_endpt.wav', 'CR4A_endpt.wav', 'DN4B_endpt.wav', 'EA4A_endpt.wav', 'EH4B_endpt.wav', 'EI4B_endpt.wav', 'EK4B_endpt.wav', 'FD4B_endpt.wav', 'FI4A_endpt.wav', 'FJ4A_endpt.wav', 'FK4B_endpt.wav', 'GG4A_endpt.wav', 'HA4A_endpt.wav', 'HH4A_endpt.wav', 'IG4A_endpt.wav', 'IH4A_endpt.wav', 'IN4B_endpt.wav', 'KR4A_endpt.wav', 'KT4A_endpt.wav', 'LS4B_endpt.wav', 'MK4A_endpt.wav', 'NC4A_endpt.wav', 'NH4B_endpt.wav', 'SP4B_endpt.wav'], '7': ['AJ7A_endpt.wav', 'AN7A_endpt.wav', 'BD7A_endpt.wav', 'CB7A_endpt.wav', 'DG7A_endpt.wav', 'DG7B_endpt.wav', 'DL7A_endpt.wav', 'EA7A_endpt.wav', 'EE7A_endpt.wav', 'EG7A_endpt.wav', 'EG7B_endpt.wav', 'EI7B_endpt.wav', 'ES7A_endpt.wav', 'FC7B_endpt.wav', 'FF7A_endpt.wav', 'FK7A_endpt.wav', 'FK7B_endpt.wav', 'FL7B_endpt.wav', 'GT7B_endpt.wav', 'HL7B_endpt.wav', 'HP7B_endpt.wav', 'IE7A_endpt.wav', 'IF7B_endpt.wav', 'JJ7A_endpt.wav', 'JP7A_endpt.wav', 'JT7B_endpt.wav', 'KD7B_endpt.wav', 'LD7B_endpt.wav', 'NC7B_endpt.wav', 'SP7A_endpt.wav'], '2': ['AW2A_endpt.wav', 'AW2B_endpt.wav', 'BH2A_endpt.wav', 'CF2A_endpt.wav', 'CF2B_endpt.wav', 'CL2B_endpt.wav', 'DL2A_endpt.wav', 'DN2A_endpt.wav', 'EA2B_endpt.wav', 'EH2B_endpt.wav', 'ES2B_endpt.wav', 'FF2B_endpt.wav', 'FJ2A_endpt.wav', 'FJ2B_endpt.wav', 'FK2A_endpt.wav', 'FK2B_endpt.wav', 'HN2A_endpt.wav', 'IF2B_endpt.wav', 'IG2A_endpt.wav', 'KC2A_endpt.wav', 'KF2A_endpt.wav', 'KH2A_endpt.wav', 'KR2A_endpt.wav', 'MK2B_endpt.wav', 'MS2A_endpt.wav', 'NH2B_endpt.wav', 'PE2B_endpt.wav', 'PK2A_endpt.wav', 'PK2B_endpt.wav', 'SP2B_endpt.wav'], 'Z': ['ACZB_endpt.wav', 'AIZA_endpt.wav', 'AJZB_endpt.wav', 'AWZA_endpt.wav', 'BDZA_endpt.wav', 'BIZA_endpt.wav', 'CFZB_endpt.wav', 'CGZB_endpt.wav', 'CMZA_endpt.wav', 'DLZB_endpt.wav', 'EAZA_endpt.wav', 'ECZA_endpt.wav', 'FIZA_endpt.wav', 'FJZB_endpt.wav', 'FLZA_endpt.wav', 'GTZA_endpt.wav', 'HAZB_endpt.wav', 'HSZA_endpt.wav', 'IHZA_endpt.wav', 'JNZA_endpt.wav', 'JPZB_endpt.wav', 'JRZA_endpt.wav', 'JRZB_endpt.wav', 'KDZA_endpt.wav', 'KDZB_endpt.wav', 'MKZA_endpt.wav', 'NCZB_endpt.wav', 'NHZA_endpt.wav', 'RAZB_endpt.wav', 'RSZB_endpt.wav'], '5': ['AE5B_endpt.wav', 'AI5A_endpt.wav', 'AL5A_endpt.wav', 'AL5B_endpt.wav', 'BD5B_endpt.wav', 'EA5A_endpt.wav', 'EG5A_endpt.wav', 'ES5A_endpt.wav', 'FI5A_endpt.wav', 'FJ5A_endpt.wav', 'GR5A_endpt.wav', 'GR5B_endpt.wav', 'GT5A_endpt.wav', 'HA5A_endpt.wav', 'HG5A_endpt.wav', 'HN5A_endpt.wav', 'IE5A_endpt.wav', 'IE5B_endpt.wav', 'IF5B_endpt.wav', 'IL5B_endpt.wav', 'JC5B_endpt.wav', 'JP5B_endpt.wav', 'KN5A_endpt.wav', 'KR5B_endpt.wav', 'LA5A_endpt.wav', 'LD5B_endpt.wav', 'LS5B_endpt.wav', 'MW5A_endpt.wav', 'NG5A_endpt.wav', 'RN5A_endpt.wav']}
mat_list = []
for utt in dict_target2utt[target]:  # utt   AJ7A_endpt.wav
    print("utt ", utt)
    ark = dict_utt2feat[utt]  # 对应的特征     train/feats.ark:112443
    print(" ark  ", ark)
    mat = kaldi_io.read_mat(ark)
    print("mat", mat)
    print("mat.shape", mat.shape)   # (49, 39)  (61, 39)   (54, 39)......
    mat_list.append(mat)
    print(" mat_list", mat_list)    # (1, 49, 39)  (2,) (3,)  .....(30,)  30句话
    print(" mat_list.shape", np.array(mat_list).shape)
print("mat", mat)
print("mat,shape", mat.shape)
print(" mat_list", mat_list)
print(" mat_list.shape", np.array(mat_list).shape)
np.concatenate(mat_list, axis=0)   # (1771, 39)
print(" np.concatenate(mat_list, axis=0)", np.concatenate(mat_list, axis=0))
print(" np.concatenate(mat_list, axis=0).shape", np.concatenate(mat_list, axis=0).shape)

gmms = {}  # 字典
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
for target in targets:
    gmms[target] = 3 # dict{str:gmm} 类的实例化  Initial model   为每一个类型数据创建一个GMM模型，保存在gmms(字典)
    print("target", target)
    print("gmms[target]", gmms[target])
    print("gmms", gmms)