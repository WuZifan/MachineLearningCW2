# # -*- coding:utf8 -*-
import scipy.io as sio
import math as mt
import matplotlib.pyplot as plt
import numpy as np
import time

# from graphviz import Digraph

__Author__ = 'Tree_Diagram'

if __name__ == "__main__":
    # 导入数据
    # matfn = u'cleandata_students.mat'
    matfn = u'noisydata_students.mat'

    data = sio.loadmat(matfn)


    res = data['y']
    element = data['x']

    print np.where(element > 5)
    for ind, val in enumerate(res):
        temp = filter(lambda x: x > 6 or x <= 0, val)
        if len(temp) > 0:
            print ind, temp

    # for ind, val in enumerate(element):
    #     temp = filter(lambda x: x != 1 or x != 2, val)
    #     if len(temp) > 0:
    #         print ind, temp

