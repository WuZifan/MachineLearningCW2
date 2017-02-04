# # -*- coding:utf8 -*-
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
matfn = u'/home/roland/PycharmProjects/test1/forStudents/cleandata_students.mat'
data = sio.loadmat(matfn)
# 45个属性的数据
facial_expression=data['y']
# 不同的label
actions=data['x']

print len(data['x'][0])

def CHOOSE_EMOTION(facial_expression,emotion):
    choosen_emotion=[]
    for emo in facial_expression:
        if emo==emotion:
            choosen_emotion.append(1)
        else:
            choosen_emotion.append(0)
    return choosen_emotion

def generate_attributes(num_attributes):
    return [range(0,45)]

def DECISION_TREE_LEARNING(examples,attributes,binary_targets):
    tree=1
    return tree