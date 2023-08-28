from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import dgl
from tkinter import _flatten



def update_parent_features(label_network:dgl.DGLGraph, labels):
    # 获取图中的所有边
    edges = label_network.edges()
    # 对于图中的每条边
    for child_idx, parent_idx in zip(edges[0], edges[1]):
        # 如果child节点的特征值大于parent节点的特征值
        if labels[0][child_idx] > labels[0][parent_idx]:
            # 更新parent节点的特征值为child节点的特征值
            labels[0][parent_idx] = labels[0][child_idx]
        # 更新labels的第二列为second_dim_elements
    return labels

def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = 0.06
    aupr += metrics.auc(recall, precision)
    return aupr

def calculate_performance(actual, pred_prob, label_network:dgl.DGLGraph, threshold=0.2, average='micro'):
    pred_lable = []
    actual_label = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int)
        eachline = eachline.tolist()
        # eachline = update_parent_features(label_network,eachline)
        pred_lable.append(list(_flatten(eachline)))
    for l in range(len(actual)):
        eachline = (np.array(actual[l])).astype(np.int)
        eachline = eachline.tolist()
        actual_label.append(list(_flatten(eachline)))
    f_score = f1_score(actual_label, pred_lable, average=average)
    recall = recall_score(actual_label, pred_lable, average=average)
    precision = precision_score(actual_label,  pred_lable, average=average)
    return f_score, precision, recall
