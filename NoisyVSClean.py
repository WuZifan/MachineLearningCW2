# # -*- coding:utf8 -*-
import scipy.io as sio
import math as mt
import matplotlib.pyplot as plt
import numpy as np
import time

# from graphviz import Digraph

__Author__ = 'Tree_Diagram'


def topythonlist(data):
    mylist = []
    for d in data:
        for dd in d:
            mylist.append(dd)
    return mylist


def topythonnestedlist(data):
    mynestedlist = []
    for da in data:
        mylist = []
        for dda in da:
            mylist.append(dda)
        mynestedlist.append(mylist)
    return mynestedlist

# 比较连个例子是否相同
def compareExample(cl,nl):
    for i in range(0,len(cl)):
        if cl[i]!=nl[i]:
            return False
    return True

def count_every_attribute(mylist):
    pa_result = []
    for sub_cl in mylist:
        su_result = []
        for sub_index in range(0, 45):
            sum_result = 0
            for sc in sub_cl:
                sum_result = sum_result + sc[1][sub_index]
            su_result.append(sum_result)
        pa_result.append(su_result)
    return pa_result


def divideByLabbel(labbel,examples):
    clean_list = []
    for i in range(1, 7):
        save_list = []
        for ind, val in enumerate(labbel):
            if val == i:
                save_list.append([val, examples[ind]])
        clean_list.append(save_list)
    return clean_list

if __name__ == "__main__":
    # 导入数据
    matfn_clean = u'cleandata_students.mat'
    matfn_noisy = u'noisydata_students.mat'

    clean_data = sio.loadmat(matfn_clean)
    noisy_data = sio.loadmat(matfn_noisy)

    # 45个属性的数据,对应choose_emotion中第一个参数
    clean_labbel = topythonlist(clean_data['y'])

    # 不同的label,对应examples
    clean_examples = topythonnestedlist(clean_data['x'])

    # noisy
    noisy_labbel = topythonlist(noisy_data['y'])
    noisy_examples = topythonnestedlist(noisy_data['x'])

    clean_list=divideByLabbel(clean_labbel,clean_examples)
    noisy_list=divideByLabbel(noisy_labbel,noisy_examples)
    for kk in clean_list:
        print str(kk[0][0])+" "+str(len(kk))

    print ""

    for kk in noisy_list:
        print str(kk[0][0])+" "+str(len(kk))

    clean_result_list=count_every_attribute(clean_list)
    noisy_result_list=count_every_attribute(noisy_list)

    for x in range(0,6):
        result=[]
        for y in range(0,45):
            result.append(abs(noisy_result_list[x][y]-clean_result_list[x][y]))
        print result
        print str(result.index(max(result)))+" "+str(max(result))


    '''
    result=[]
    for ce in noisy_examples:
        for ne in noisy_examples:
            flag=compareExample(ce,ne)
            if flag:
                ce_index=noisy_examples.index(ce)
                ne_index=noisy_examples.index(ne)
                if noisy_labbel[ne_index]!=noisy_labbel[ce_index]:
                    result.append([ce_index,ne_index])

    print result
    '''
