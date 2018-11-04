import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = None

nhts = pd.read_csv('DAYV2PUB.csv')


####################################
from collections import Counter
from collections import defaultdict

import sys
import string
import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy import linalg as LA
import csv

from random import randrange


def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )

    return dataset

def get_most_commons(dataset, skip=100, total=100):
    counter = Counter()
    for item in dataset:
        counter = counter + Counter(set(item[1]))

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words



def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        # vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)


def vec_to_df(vec):
    df = DataFrame(vec, columns=['label', 'x'])
    return df

def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)

#################################SVM part
def svm(orig_features, labels):
    # test sub-gradient SVM
    features = np.hstack((orig_features, np.ones((orig_features.shape[0], 1))))

    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1

    return w

def svm_pred(w, orig_features):
    features = np.hstack((orig_features, np.ones((orig_features.shape[0], 1))))
    return np.where((features @ w) >= 0, 1, 0)

########################### DT part
def gini(vector):
    result = 1
    value, num = np.unique(vector, return_counts=True)
    p_x = num.astype('float')/len(vector)
    for p in p_x:
        result -= p * p
    return result

def gini_gain(feature, label):
    rowNum, colNum = feature.shape
    gini_s = gini(label)
    gain = zeros(colNum)
    for i in range(colNum):
        val, count = np.unique(feature[:, i], return_counts=True)
        for j in range(len(val)):
            freq =  count[j]/float(rowNum)
            gain[i] += freq * gini(label[feature[:,i]==val[j]])
        gain[i] = gini_s - gain[i]
    return argsort(-gain)[0]



def end_node_result(label):
    counts = np.bincount(label)
    return np.argmax(counts)



def data_split(feature, label, index, value):
    sub_feature = list()
    sub_label = list()

    for i in range(len(label)):
        if feature[i][index] == value:
            sub_feature.append(feature[i])
            sub_label.append(label[i])
    return np.array(sub_feature), np.array(sub_label)

def build_tree(feature, label, depth_limit, example_limit, depth):
    if gini(label) == 0.0:
        return label[0]
    if len(feature[0]) == 1:
        return end_node_result(label)
    if depth >= depth_limit:
        return end_node_result(label)
    if len(label) <= example_limit:
        return end_node_result(label)

    ind = gini_gain(feature, label)
    tree = {ind:{}}
    f_values = feature[:, ind]
    val_set = set(f_values)
    for v in val_set:
        new_f, new_l = data_split(feature, label, ind, v)
        tree[ind][v] = build_tree(new_f, new_l, depth_limit, example_limit, depth+1)
    return tree

def predict(tree, f_vec):
    first_node = list(tree.keys())[0]
    next_dic = tree[first_node]
    global pred_val

    for key in next_dic.keys():
        if f_vec[first_node] == key:
            if isinstance(next_dic[key], dict):
                pred_val = predict(next_dic[key], f_vec)
            else:
                pred_val = next_dic[key]
    return pred_val


def DT_predict(tree, test_f, test_l):
    pred_label = []
    for record in test_f:
        pred = predict(tree, record)
        pred_label.append(pred)
    pred_array = np.array(pred_label)

    return calc_error(pred_array, test_l)

def predict_vec(tree, test_f): ####generate prediction result
    pred_label = []
    for record in test_f:
        pred = predict(tree, record)
        pred_label.append(pred)
    pred_array = np.array(pred_label)

    return pred_array

def bag_sample(feature, label):
    new_f = []
    new_l = []
    while len(new_l) < len(label):
        pos = random.randrange(len(label))
        new_f.append(feature[pos])
        new_l.append(label[pos])
    return np.array(new_f), np.array(new_l)

def bag_single(tree_list, row):
    predictions = [predict(tree, row) for tree in tree_list]
    return max(set(predictions), key=predictions.count)

def bag(train_f, train_l, test_f, test_l, M, depth_limit):
    tree_list = []
    predictions = []
    for i in range(M):
        new_f, new_l = bag_sample(train_f, train_l)
        tree = build_tree(new_f, new_l, depth_limit, 10, 0)
        tree_list.append(tree)
    for row in test_f:
        predictions.append(bag_single(tree_list, row))
    pred_array = np.array(predictions)
    return calc_error(pred_array, test_l)

# def RF_gini_gain(feature, label, f_list):
#     rowNum, colNum = feature.shape
#     gini_s = gini(label)
#     gain = zeros(colNum)
#     for i in f_list:
#         val, count = np.unique(feature[:, i], return_counts=True)
#         for j in range(len(val)):
#             freq =  count[j]/float(rowNum)
#             gain[i] += freq * gini(label[feature[:,i]==val[j]])
#         gain[i] = gini_s - gain[i]
#     return argsort(-gain)[0]
#
# def RF_build_tree(feature, label, depth_limit, example_limit, depth):
#     if gini(label) == 0.0:
#         return label[0]
#     if len(feature[0]) == 1:
#         return end_node_result(label)
#     if depth >= depth_limit:
#         return end_node_result(label)
#     if len(label) <= example_limit:
#         return end_node_result(label)
#
#     num_f = int(sqrt(len(feature[0])))
#
#     f_list = random.sample(range(len(feature[0])), num_f)
#
#
#     ind = RF_gini_gain(feature, label, f_list)
#     print(ind)
#     tree = {ind:{}}
#     f_values = feature[:, ind]
#     val_set = set(f_values)
#     for v in val_set:
#         new_f, new_l = data_split(feature, label, ind, v)
#         tree[ind][v] = RF_build_tree(new_f, new_l, depth_limit, example_limit, depth+1)
#     return tree
#
# def RF(train_f, train_l, test_f, test_l, M, depth_limit):
#     tree_list = []
#     predictions = []
#     for i in range(M):
#         new_f, new_l = bag_sample(train_f, train_l)
#         tree = RF_build_tree(new_f, new_l, depth_limit, 10, 0)
#         tree_list.append(tree)
#     for row in test_f:
#         predictions.append(bag_single(tree_list, row))
#     pred_array = np.array(predictions)
#     return calc_error(pred_array, test_l)

def RF_gini_gain(feature, label, k):
    rowNum, colNum = feature.shape
    gini_s = gini(label)
    gain = zeros(colNum) ############################################????
    f_ind_list = []
    while len(f_ind_list) < k:
        ind = randrange(len(feature[0]))
        if ind not in f_ind_list:
            f_ind_list.append(ind)
    for i in f_ind_list:
        val, count = np.unique(feature[:, i], return_counts=True)
        for j in range(len(val)):
            freq =  count[j]/float(rowNum)
            gain[i] += freq * gini(label[feature[:,i]==val[j]])
        gain[i] = gini_s - gain[i]
    return argsort(-gain)[0]


def RF_build_tree(feature, label, depth_limit, example_limit, depth, k):
    if gini(label) == 0.0:
        return label[0]
    if len(feature[0]) == 1:
        return end_node_result(label)
    if depth >= depth_limit:
        return end_node_result(label)
    if len(label) <= example_limit:
        return end_node_result(label)

    ind = RF_gini_gain(feature, label, k)
    tree = {ind:{}}
    f_values = feature[:, ind]
    val_set = set(f_values)
    for v in val_set:
        new_f, new_l = data_split(feature, label, ind, v)
        tree[ind][v] = RF_build_tree(new_f, new_l, depth_limit, example_limit, depth+1, k)
    return tree

def RF_predict(tree_list, row):
    predictions = [predict(tree, row) for tree in tree_list]
    return max(set(predictions), key=predictions.count)


def RF(train_f, train_l, test_f, test_l, depth_limit, M, k):
    tree_list = []
    for i in range(M):
        new_f, new_l = bag_sample(train_f, train_l)
        tree = RF_build_tree(new_f, new_l, depth_limit, 10, 0, k)
        tree_list.append(tree)
    predictions = [RF_predict(tree_list, row) for row in test_f]
    pred_array = np.array(predictions)
    loss = calc_error(pred_array, test_l)
    return loss

#########################################

def weighted_pick(weights):
    total = 0
    index_out = 0
    for i, w in enumerate(weights):
        total += w
        if random.random() * total < w:
            index_out = i
    return index_out

def weighted_sample(feature, label, weights):
    new_f = []
    new_l = []
    while len(new_l) < len(label):
        pos = weighted_pick(weights)
        new_f.append(feature[pos])
        new_l.append(label[pos])
    return np.array(new_f), np.array(new_l)

def weighted_gini(vector, weights):
    pos_sum = 0
    neg_sum = 0

    for i in range(len(vector)):
        if vector[i] == 1:
            pos_sum += weights[i]
        else:
            neg_sum += weights[i]
    p_pos = pos_sum/sum(weights)
    p_neg = neg_sum/sum(weights)
    result = 1 - p_pos * p_pos - p_neg * p_neg
    return result

def weighted_gini_gain(feature, label, weights):
    rowNum, colNum = feature.shape

    gini_s = weighted_gini(label, weights)

    gain = zeros(colNum)
    for i in range(colNum):
        val, count = np.unique(feature[:, i], return_counts=True)
        for j in range(len(val)):
            freq = sum(weights[feature[:,i]==val[j]])/sum(weights)
            gain[i] += freq * weighted_gini(label[feature[:,i]==val[j]], weights[feature[:,i]==val[j]])
        gain[i] = gini_s - gain[i]
    return argsort(-gain)[0]

def weighted_data_split(feature, label, index, value, weights):
    sub_feature = list()
    sub_label = list()
    sub_weights = list()

    for i in range(len(label)):
        if feature[i][index] == value:
            sub_feature.append(feature[i])
            sub_label.append(label[i])
            sub_weights.append(weights[i])
    return np.array(sub_feature), np.array(sub_label), np.array(sub_weights)

def boost_build_tree(feature, label, depth_limit, example_limit, depth, weights):
    if gini(label) == 0.0:
        return label[0]
    if len(feature[0]) == 1:
        return end_node_result(label)
    if depth >= depth_limit:
        return end_node_result(label)
    if len(label) <= example_limit:
        return end_node_result(label)


    ind = weighted_gini_gain(feature, label, weights)
    tree = {ind:{}}
    f_values = feature[:, ind]
    val_set = set(f_values)
    for v in val_set:
        new_f, new_l, new_weights = weighted_data_split(feature, label, ind, v, weights)
        tree[ind][v] = boost_build_tree(new_f, new_l, depth_limit, example_limit, depth+1, new_weights)
    return tree

def boosted_single(tree_list, row, stages):
    pred_result = []
    for tree in tree_list:
        pred_result.append(predict(tree, row))
    pred_array = np.array(pred_result)
    neg_array = ones(len(pred_array))
    for i in range(len(pred_array)):
        if pred_array[i] == 0:
            neg_array[i] = -1

    wtd_avg = sum(stages * neg_array) / sum(stages)
    if wtd_avg >= 0:
        prediction = 1
    else:
        prediction = 0
    return prediction

def normalize(x):
    norm_x = x/sum(x)
    return norm_x


#################################################### version 2
def boosted_DT(train_f, train_l, test_f, test_l, M, depth_limit):
    tree_list = []
    predictions = []
    # weights = [1] * len(train_l)
    weights = np.ones(len(train_l))/len(train_l)
    stages = []
    wtd_error = 1
    for i in range(M):
        # if wtd_error == 0:
        #     break
        # new_f, new_l = weighted_sample(train_f, train_l, weights)
        # tree = boost_build_tree(new_f, new_l, depth_limit, 10, 0, weights)
        tree = boost_build_tree(train_f, train_l, depth_limit, 10, 0, weights)
        # tree = build_tree(new_f, new_l, depth_limit, 10, 0)
        # print('round')
        # print(i)
        tree_list.append(tree)
        # pred_result = predict_vec(tree, new_f)
        # dif = np.absolute(pred_result - new_l)
        pred_result = predict_vec(tree, train_f)
        dif = np.absolute(pred_result - train_l)
        # print(sum(dif))
        wtd_error = sum(weights*dif)
        # print(wtd_error)
        stage = float(np.log((1-wtd_error)/max(wtd_error, 1e-16)) )  # error may become 0
        # print(stage)
        weights = normalize(weights * np.exp(stage * dif))
        # print(max(weights))
        stages.append(stage)
        if wtd_error == 0:
            break

    for row in test_f:
        predictions.append(boosted_single(tree_list, row, stages))
    pred_array = np.array(predictions)
    return calc_error(pred_array, test_l)




train_data = read_dataset('yelp_train0.txt')
test_data = read_dataset('yelp_test0.txt')
common_words = get_most_commons(train_data, skip=100, total=1000)
train_f, train_l = generate_vectors(train_data, common_words)
test_f, test_l = generate_vectors(test_data, common_words)


    #
# train_f0 = np.array([[0,1,0,1],[1,1,0,1],[0,1,1,0], [0,1,1,1],[0,1,1,0]])
# train_l0 = np.array([0,1,1,1,0])
# train_l1 = np.array([7,1,8,1,9])

# print('mine')
# w = svm(train_f, train_l)
# test_pred = svm_pred(w, test_f)
# print('ZERO-ONE-LOSS-SVM', calc_error(test_pred, test_l))

# tree = build_tree(train_f, train_l, 10, 10, 0)
#
# DT_loss = DT_predict(tree,test_f,test_l)
# print(DT_loss)
#
BT_loss = bag(train_f, train_l, test_f, test_l, 50, 10)

print(BT_loss)
#
# k = int(sqrt(len(train_f[0])))
#
# RF_loss = RF(train_f, train_l, test_f, test_l, 10, 50, k)
#
# print(RF_loss)
#
#
# loss_boost = boosted_DT(train_f, train_l, test_f, test_l, 50, 10)
#
# print(loss_boost)
#
# print('skl')
#
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import AdaBoostClassifier
#
# clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=10)
#
# clf = clf.fit(train_f, train_l)
#
# skpredict= clf.predict(test_f)
#
# sk_dtloss = calc_error(skpredict, test_l)
#
# print(sk_dtloss)
# #
# #
# skbag = BaggingClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=10), n_estimators=50)
# skbag = skbag.fit(train_f, train_l)
# skbagpred = skbag.predict(test_f)
# skbagloss = calc_error(skbagpred, test_l)
# print(skbagloss)
#
#
# # skrf = RandomForestClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=10), n_estimators=50)
# skrf = RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators=50)
# skrfit=skrf.fit(train_f, train_l)
# skrfpred = skrfit.predict(test_f)
# skrfloss = calc_error(skrfpred, test_l)
# print(skrfloss)
#
# skbst = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, min_samples_split=10), n_estimators=50)
# skbstfit = skbst.fit(train_f, train_l)
# skbstpred = skbstfit.predict(test_f)
# skbstloss = calc_error(skbstpred, test_l)
# print(skbstloss)




##############################################################################


# if __name__ == '__main__':
#     if len(sys.argv) == 4:
#         train_data_file = sys.argv[1]
#         test_data_file = sys.argv[2]
#         model_idx = int(sys.argv[3])
#
#         train_data = read_dataset(train_data_file)
#         test_data = read_dataset(test_data_file)
#
#         common_words = get_most_commons(train_data, skip=100, total=4000)
#
#         train_f, train_l = generate_vectors(train_data, common_words)
#         test_f, test_l = generate_vectors(test_data, common_words)
#
#         if model_idx == 1:
#             tree = build_tree(train_f, train_l, 10, 10, 0)
#             loss = DT_predict(tree, test_f, test_l)
#             print('ZERO-ONE-LOSS-DT', loss)
#         elif model_idx == 2:
#             loss = bag(train_f, train_l, test_f, test_l, 50, 10)
#             print('ZERO-ONE-LOSS-BT', loss)
#         elif model_idx == 3:
#             loss = RF(train_f, train_l, test_f, test_l, 50, 10)
#             print('ZERO-ONE-LOSS-RF', loss)
#         else:
#             print('Illegal modelIdx')
#     else:
#         print('usage: python hw3.py train.csv test.csv modelIdx')


#################### Analysis part 1 ###################################################
#
# data = read_dataset('yelp_data.csv')
# random.shuffle(data)
# data_list = []
# for i in range(10):
#     S = data[i*200:(i+1)*200]
#     data_list.append(S)
#
# out_dic_dt = dict()
# out_dic_bt = dict()
# out_dic_rf = dict()
# out_dic_svm = dict()
# out_dic_bst = dict()
#
# for percentage in [0.025, 0.05, 0.125, 0.25]:
#     TSS = percentage * 2000
#     in_dic_dt = dict()
#     in_dic_bt = dict()
#     in_dic_rf = dict()
#     in_dic_svm = dict()
#     in_dic_bst = dict()
#
#     for idx in range(10):
#         SC = []
#         test_set = data_list[idx]
#         for i_in in range(10):
#             if i_in != idx:
#                 SC.extend(data_list[i_in])
#         random.shuffle(SC)
#         train_set = SC[:int(TSS)]
#         my_list = []
#         for item in train_set:
#             my_list += list(item[1])
#         counter = Counter(my_list)
#         word_num = min(len(counter), 1000)
#         skip = 100
#         temp = counter.most_common(word_num + skip)[skip:]
#         common_words = [item[0] for item in temp]
#         train_f, train_l = generate_vectors(train_set, common_words)
#         test_f, test_l = generate_vectors(test_set, common_words)
#
        # DT part
        # tree = build_tree(train_f, train_l, 10, 10, 0)
        # in_dic_dt[idx] = DT_predict(tree,test_f,test_l)
        # # BT part
        # in_dic_bt[idx] = bag(train_f, train_l, test_f, test_l, 50, 10)
        # # RF part
        # in_dic_rf[idx] = RF(train_f, train_l, test_f, test_l, 50, 10)
        # # SVM part
        # w = svm(train_f, train_l)
        # test_pred = svm_pred(w, test_f)
        # in_dic_svm[idx] = calc_error(test_pred, test_l)
        # # boost part
        # in_dic_bst[idx] = boosted_DT(train_f, train_l, test_f, test_l, 50, 10)
#     out_dic_dt[percentage] = in_dic_dt
#     out_dic_bt[percentage] = in_dic_bt
#     out_dic_rf[percentage] = in_dic_rf
#     out_dic_svm[percentage] = in_dic_svm
#     out_dic_bst[percentage] = in_dic_bst
# df_dt = DataFrame(out_dic_dt)
# df_bt = DataFrame(out_dic_bt)
# df_rf = DataFrame(out_dic_rf)
# df_svm = DataFrame(out_dic_svm)
# df_bst = DataFrame(out_dic_bst)
#
# # print('avg. of zero-one loss in NBC')
# # print(df_nbc.mean())
# # print('Standard error of zero-one loss in NBC')
# # print(df_nbc.std()/(10**(1/2.0)))
# # print('avg. of zero-one loss in LR')
# # print(df_lr.mean())
# # print('Standard error of zero-one loss in LR')
# # print(df_lr.std()/(10**(1/2.0)))
# # print('avg. of zero-one loss in SVM')
# # print(df_svm.mean())
# # print('Standard error of zero-one loss in SVM')
# # print(df_svm.std()/(10**(1/2.0)))
#
# fig = plt.figure()
# part1 = plt.axes()
# plt.errorbar(df_dt.columns, df_dt.mean(), df_dt.std()/(10**(1/2.0)), color='r', linestyle='solid', linewidth=1.0, label='DT')
# plt.errorbar(df_bt.columns, df_bt.mean(), df_bt.std()/(10**(1/2.0)), color = 'g', linestyle=':', linewidth=1.0, label='BT')
# plt.errorbar(df_rf.columns, df_rf.mean(), df_rf.std()/(10**(1/2.0)), color = 'b', linestyle='-.', linewidth=1.0, label='RF')
# plt.errorbar(df_svm.columns, df_svm.mean(), df_svm.std()/(10**(1/2.0)), color='black', linestyle='--', linewidth=1.0, label='SVM')
# plt.errorbar(df_bst.columns, df_bst.mean(), df_bst.std()/(10**(1/2.0)), color = 'm', linestyle=':', linewidth=1.0, label='BOOST')
# part1.set_xlabel('TSS percentage')
# part1.set_ylabel('Zero-one loss')
#
# part1.legend(loc='best')
# part1.set_title('Performance of five models')
#
# plt.show()
#
# ###### write result to a csv file
# writer = pd.ExcelWriter('output_part1.xlsx', engine='xlsxwriter')
# df_dt.to_excel(writer, 'dt')
# df_bt.to_excel(writer, 'bt')
# df_rf.to_excel(writer, 'rf')
# df_svm.to_excel(writer, 'SVM')
# df_bst.to_excel(writer, 'boost')
# writer.save()