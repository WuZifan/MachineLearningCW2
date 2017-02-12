# # -*- coding:utf8 -*-
import scipy.io as sio
import math as mt
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import hashlib
import graphviz
from graphviz import Digraph

__Author__ = 'Tree_Diagram'


# 将选中的表情设置为1，其他为0,对应binary_targets
def choose_emotion(facial_expression, emotion):
    choosen_emotion = []
    for emo in facial_expression:
        if emo == emotion:
            choosen_emotion.append(1)
        else:
            choosen_emotion.append(0)
    return choosen_emotion


# 将属性编号为0到44，对应attributes
def generate_attributes(num_attributes):
    return range(1, num_attributes + 1)


def examples_havesamevalue(binary_targets):
    flag = True
    if len(binary_targets) != 0:
        target = binary_targets[0]

    # 遍历所有的example
    for j in range(0, len(binary_targets)):
        if target != binary_targets[j]:
            flag = False
            break
    if flag:
        return target
    else:
        return -1


def get_information_gain(p, n):
    pf = float(p)
    nf = float(n)
    return - pf / (pf + nf) * mt.log(pf / (pf + nf), 2) \
           - nf / (pf + nf) * mt.log(nf / (pf + nf), 2)


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

        for ind, value in enumerate(data_set):
            if value[index] == 1:
                if binary_target[ind] == 1:
                    pn1 += 1
                else:
                    nn1 += 1
            else:
                if binary_target[ind] == 1:
                    pn0 += 1
                else:
                    nn0 += 1

        if pn1 == 0 or nn1 == 0:
            entropy1 = 0
        else:
            entropy1 = float(pn1 + nn1) / float(n1 + n0) * get_information_gain(pn1, nn1)

        if pn0 == 0 or nn0 == 0:
            entropy0 = 0
        else:
            entropy0 = float(pn0 + nn0) / float(n1 + n0) * get_information_gain(pn0, nn0)

        information_gain.append(entropy - entropy0 - entropy1)

    return information_gain.index(max(information_gain))


def majority_value(binary_targets):
    length = 0
    for row in binary_targets:
        if row == 1:
            length = length + 1

    if length * 2 >= len(binary_targets):
        return 1
    else:
        return 0


def generate_sub(examples, binary_targets, best_attribute, attribute_state):
    myexamples = []
    mybinary_targets = []
    for ind, val in enumerate(examples):
        if examples[ind][best_attribute] == attribute_state:
            myexamples.append(examples[ind])
            mybinary_targets.append(binary_targets[ind])
    return myexamples, mybinary_targets


# 主要被调用函数
TREE_NODES = []
myNAME=1
def DECISION_TREE_LEARNING(examples, attributes, binary_targets):
    global myNAME
    target_value = examples_havesamevalue(binary_targets)
    if target_value != -1:
        if target_value == 1:
            node = [hashlib.sha256(str(myNAME)).hexdigest(),'YES', []]
            myNAME+=1
            TREE_NODES.append(node)
            return node
        else:
            node = [hashlib.sha256(str(myNAME)).hexdigest(), 'NO', []]
            myNAME+=1
            TREE_NODES.append(node)
            return node
    elif len(attributes) == 0:
        ma_value = majority_value(binary_targets)
        if ma_value == 1:
            node = [hashlib.sha256(str(myNAME)).hexdigest(), 'YES', []]
            myNAME+=1
            TREE_NODES.append(node)
            return node
        else:
            node = [hashlib.sha256(str(myNAME)).hexdigest(), 'NO', []]
            myNAME+=1
            TREE_NODES.append(node)
            return node
    else:
        best_attribute = choose_best_attribute(examples, attributes, binary_targets)
        tree = [hashlib.sha256(str(myNAME)).hexdigest(), str(attributes[best_attribute]), []]
        myNAME+=1
        for attribute_state in [0, 1]:
            newexamples, newbinary_targets = generate_sub(examples, binary_targets, best_attribute, attribute_state)
            if len(newexamples) == 0:
                ma_value2 = majority_value(binary_targets)
                if ma_value2 == 1:
                    node = [hashlib.sha256(str(myNAME)).hexdigest(), 'YES', []]
                    myNAME+=1
                    TREE_NODES.append(node)
                    return node
                else:
                    node = [hashlib.sha256(str(myNAME)).hexdigest(), 'NO', []]
                    myNAME+=1
                    TREE_NODES.append(node)
                    return node
            else:
                newattributes = attributes[:best_attribute] + attributes[best_attribute + 1:]
                newexamples = map(lambda x: x[:best_attribute] + x[best_attribute + 1:], newexamples)
                #print "best attribute:"+str(best_attribute+1)+"state: "+str(attribute_state)
                #print "binary: "+str(newbinary_targets)
                nextTreeNode = DECISION_TREE_LEARNING(newexamples, newattributes, newbinary_targets)
                tree[2].append(nextTreeNode[0])
        TREE_NODES.append(tree)
    return tree

def DrawDecisionTree(label, tree, dot):
    item = []
    for node in tree:
        if node[0] == label:
            item = node
            break
    [nodelabel, name, leaves]= item
    dot.node(nodelabel, str(name))
    if len(leaves) == 0:
        pass
    else:
        DrawDecisionTree(leaves[0], tree, dot)
        DrawDecisionTree(leaves[1], tree, dot)
        dot.edge(nodelabel, leaves[0], label='0',_attributes=None)
        dot.edge(nodelabel, leaves[1], label='1',_attributes=None)
    return dot



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

def update_comment_result(comment_tree,real_result,test_label):
    myresult=-1
    myweight=0-sys.maxint
    # 选出这次的label
    for ind in range(0,6):
        if test_label[ind]==1:
            if myweight <= comment_tree[ind]:
                myweight=comment_tree[ind]
                myresult=ind+1

    # 更新comment
    for ind in range(0,6):
        if ind+1==real_result:
            if test_label[ind]==1:
                comment_tree[ind]+=1
            else:
                comment_tree[ind]-=1
        elif ind+1!=real_result:
            if test_label[ind]==0:
                comment_tree[ind]+=1
            else:
                comment_tree[ind]-=1
    return myresult,comment_tree

# 预测函数,boost方法
def predictions_boost(TreeList, testData,testBinary):
    # 存储对每个example的6棵树的判断
    labbel = []
    # 存储对每棵树的评价,这里硬编码了
    comment_tree=[1,1,1,1,1,1]
    # 最终结果
    finalresult=[]
    for inx,exam in enumerate(testData):
        for ind, tree in enumerate(TreeList):
            root = tree[-1]
            flag,depth= find_label(exam, tree, root)
            myfalg = True
            labbel.append(flag)
        # print("labbel: "+ str(labbel))
        everyResult,comment_tree=update_comment_result(comment_tree,testBinary[inx],labbel)
        finalresult.append(everyResult)
        labbel=[]
    print "finalresult: "+str(finalresult)
    print "comment_tree: "+str(comment_tree)
    print "finalresult: "+str(finalresult)
    print "realResult: "+str(testBinary)
    return finalresult;

# 预测函数,随机
def predictions(TreeList, testData):
    labbel = []
    myfalg = False
    for exam in testData:
        for ind, tree in enumerate(TreeList):
            root = tree[-1]
            flag ,depth= find_label(exam, tree, root)
            if flag == 1:
                myfalg = True
                labbel.append(ind + 1)
                break
        if myfalg== False:
            labbel.append(-1)
        myfalg=False
    return labbel;

# 预测函数,带深度
def predictions_deepth(TreeList, testData):
    labbel = []
    myfalg = False
    for exam in testData:
        maybeRigth=[]
        for ind, tree in enumerate(TreeList):
            root = tree[-1]
            flag,deepth = find_label(exam, tree, root)
            if flag == 1:
                myfalg = True
                maybeRigth.append([ind+1,deepth])

        if myfalg == False:
            labbel.append(-1)
        else:
            final_result=0
            final_deep=0
            for vals in maybeRigth:

                if vals[1]>final_deep:
                    final_deep=vals[1]
                    final_result=vals[0]
            labbel.append(final_result)
        myfalg = False
    return labbel;

def find_label(exam, tree, root):
    if root[1] == 'YES':
        return (1,1)
    elif root[1] == 'NO':
        return (0,1)
    else:
        attribute_num = int(root[1])-1
        real_au = exam[attribute_num]
        if real_au == 0:
            next_node_index = 0;
            for ind, node in enumerate(tree):
                if node[0] == root[2][0]:
                    next_node_index = ind
                    break
            flag_label,deepth=find_label(exam, tree, tree[next_node_index])
            return (flag_label,deepth+1)
        else:
            next_node_index=0;
            for ind,node in enumerate(tree):
                if node[0]==root[2][1]:
                    next_node_index=ind
                    break
            flag_label, deepth = find_label(exam, tree, tree[next_node_index])
            return (flag_label, deepth + 1)

def load_data(path):
    data = sio.loadmat(path)
    facial_expression = topythonlist(data['y'])
    examples = topythonnestedlist(data['x'])

    return facial_expression, examples

def cross_validation_test2(examples, facial_expression):
    global TREE_NODES
    confusion_matrix_final = np.array([0] * 36).reshape(6, 6)
    test_examples = []
    train_examples = []
    test_facial_expression = []
    train_facial_expression = []
    for ind, val in enumerate(examples):
        examples[ind]
        facial_expression[ind]
    # for clean TREE
    tree_list = []
    for j in range(1, 7):
        # for binary_targets
        binary_targets = choose_emotion(facial_expression, j)
        DECISION_TREE_LEARNING(examples, attributes, binary_targets)
        tree_list.append(TREE_NODES)

        TREE_NODES = []

    test_label = predictions(tree_list, test_examples)

    confusion_matrix = np.array([0] * 36).reshape(6, 6)

    # Generate confusion matrix
    for ind, val in enumerate(test_label):
        confusion_matrix[test_facial_expression[ind] - 1, val - 1] += 1

    confusion_matrix_final = np.add(confusion_matrix_final, confusion_matrix)

    for ind, tree in enumerate(tree_list):
        dot = Digraph(comment='')
        print tree
        DrawDecisionTree(tree[-1][0], tree, dot)
        dot.render('test-output/test' + str(ind) + '.gv', view=True)

    return confusion_matrix_final

def cross_validation_test(examples, facial_expression):
    global TREE_NODES
    confusion_matrix_final = np.array([0] * 36).reshape(6, 6)

    for inx in range(0, 10):
        test_examples = []
        train_examples = []
        test_facial_expression = []
        train_facial_expression = []
        for ind, val in enumerate(examples):
            # 选取10%作为test
            if ind % 10 == inx:
                # 当选取的test数据越少，其正确越越高
                test_examples.append(examples[ind])
                test_facial_expression.append(facial_expression[ind])
            else:
                train_examples.append(examples[ind])
                train_facial_expression.append(facial_expression[ind])

        # for clean TREE
        tree_list = []

        for j in range(1, 7):
            # for binary_targets
            binary_targets = choose_emotion(train_facial_expression, j)
            DECISION_TREE_LEARNING(train_examples, attributes, binary_targets)
            tree_list.append(TREE_NODES)
            TREE_NODES = []


        #test_label = predictions_deepth(tree_list, test_examples)                      #clean 0.729083665339
        test_label = predictions_boost(tree_list,test_examples,test_facial_expression) #clean 0.73406374502
        #test_label = predictions(tree_list, test_examples)                              #clean 0.732071713147
        # time.sleep(100000)
        confusion_matrix = np.array([0] * 36).reshape(6, 6)


        # Generate confusion matrix
        for ind, val in enumerate(test_label):
            confusion_matrix[test_facial_expression[ind] - 1, val - 1] += 1

        confusion_matrix_final = np.add(confusion_matrix_final, confusion_matrix)


        for ind, tree in enumerate(tree_list):
            dot = Digraph(comment='')
            print tree
            DrawDecisionTree(tree[-1][0], tree, dot)
            dot.render('test-output/test' + str(inx)+str(ind) + '.gv', view=True)

    return confusion_matrix_final


def evaluation(confusion_matrix_final):
    average_recall = []
    average_precision_rate = []

    for goal in xrange(6):
        average_recall.append(float(confusion_matrix_final[goal, goal]) / float(confusion_matrix_final[goal].sum()))
        average_precision_rate.append(
            float(confusion_matrix_final[goal, goal]) / float(confusion_matrix_final[:, goal].sum()))

    f1_measures = []
    correct_times = 0

    for goal in xrange(6):
        f1_measures.append(2 * average_recall[goal] * average_precision_rate[goal] /
                           float(average_precision_rate[goal] + average_recall[goal]))
        correct_times += confusion_matrix_final[goal, goal]

    average_classification_rate = float(correct_times) / float(confusion_matrix_final.sum())

    print "average_recall rate: "
    print average_recall
    print "average_precision_rate: "
    print average_precision_rate
    print "f1 measure: "
    print f1_measures
    print "average classification rate: "
    print average_classification_rate


if __name__ == "__main__":
    # 导入数据
    path1 = u'cleandata_students.mat'
    path2 = u'noisydata_students.mat'

    if len(sys.argv) == 1:
        print "Empty input"
        exit(1)

    source = sys.argv[1:]

    # for attribute
    attributes = generate_attributes(45)

    for ind, path in enumerate(source):
        facial_expression, example = load_data(path)
        print "For %dth input file %s : " % (ind + 1, path)
        res = cross_validation_test(example, facial_expression)
        print "Cross Validation matrix:"
        print res
        print "evaluate result: "
        evaluation(res)
        print


    # # 45个属性的数据,对应choose_emotion中第一个参数
    # facial_expression1, examples1 = load_data(path1)
    # facial_expression2, examples2 = load_data(path2)
    #
    # print "For clean data:"
    # res = cross_validation_test(examples1, facial_expression1)
    # print "Cross Validation matrix:"
    # print res
    # print "evaluate result: "
    # evaluation(res)
    #
    # print "For noisy data:"
    # res = cross_validation_test(examples2, facial_expression2)
    # print "Cross Validation matrix:"
    # print res
    # print "evaluate result: "
    # evaluation(res)
