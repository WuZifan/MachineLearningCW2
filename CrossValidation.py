# # -*- coding:utf8 -*-
import TestDecisionTree
from LearningDecisionTree import *

__Author__ = 'Tree_Diagram'

if __name__ == "__main__":
    # 导入数据
    matfn = u'cleandata_students.mat'
    data = sio.loadmat(matfn)

    # 45个属性的数据,对应choose_emotion中第一个参数
    facial_expression = topythonlist(data['y'])

    # 不同的label,对应examples
    examples = topythonnestedlist(data['x'])

    # for attribute
    attributes = generate_attributes(45)

    test_examples = []
    train_examples = []

    test_facial_expression = []
    train_facial_expression = []

    for ind, val in enumerate(examples):
        # 选取20%作为test
        if ind % 5 == 0:
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
        tree_list.append(TestDecisionTree.TREE_NODES)
        TREE_NODES = []

    test_label = TestDecisionTree.predictions(tree_list, test_examples)
    print test_label
    print test_facial_expression

 #   for ind, val in enumerate(test_label):
