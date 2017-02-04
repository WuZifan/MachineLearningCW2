# # -*- coding:utf8 -*-
import scipy.io as sio
import math as mt
import matplotlib.pyplot as plt
import numpy as np

__Author__ = 'Tree_Diagram'

if __name__ == "__main__":
    # 导入数据
    matfn = u'/home/roland/PycharmProjects/test1/forStudents/cleandata_students.mat'
    data = sio.loadmat(matfn)
    # 45个属性的数据
    facial_expression = data['y']
    # 不同的label
    actions = data['x']

    print len(data['x'][0])


def CHOOSE_EMOTION(facial_expression, emotion):
    choosen_emotion = []
    for emo in facial_expression:
        if emo == emotion:
            choosen_emotion.append(1)
        else:
            choosen_emotion.append(0)
    return choosen_emotion


def generate_attributes(num_attributes):
    return [range(0, 45)]


def DECISION_TREE_LEARNING(examples, attributes, binary_targets):
    tree = 1
    return tree


def get_information_gain(p, n):
    return - p / (p + n) * mt.log10(p / (p + n)) / mt.log10(2) - n / (p + n) * mt.log10(n / (p + n)) / mt.log10(2)

#fine
def choose_best_attribute(data_set, attributes, binary_target):
    n0 = 0
    n1 = 0
    information_gain = []
    for value in binary_target:
        if value == 1:
            n1 += 1
        else:
            n0 += 1

    entropy = get_information_gain(n1, n0)

    num = len(data_set[0])

    for index in xrange(num):
        pn1 = 0
        pn0 = 0
        nn0 = 0
        nn1 = 0

        for ind, value in enumerate(data_set[:, index]):
            if value == 1:
                if binary_target[ind] == 1:
                    pn1 += 1
                else:
                    nn1 += 1
            else:
                if binary_target[ind] == 1:
                    pn0 += 1
                else:
                    nn0 += 1

        entropy0 = (pn0 + nn0) / (n1 + n0) * get_information_gain(pn0, nn0)
        entropy1 = (pn1 + nn1) / (n1 + n0) * get_information_gain(pn1, nn1)
        information_gain.append(entropy - entropy0 - entropy1)

    return list.index(max(information_gain))
