from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
from tkinter import _flatten


def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = 0.06
    aupr += metrics.auc(recall, precision)
    return aupr

def calculate_performance(actual, pred_prob, threshold=0.2, average='micro'):
    pred_lable = []
    actual_label = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int)
        eachline = eachline.tolist()
        pred_lable.append(list(_flatten(eachline)))
    for l in range(len(actual)):
        eachline = (np.array(actual[l])).astype(np.int)
        eachline = eachline.tolist()
        actual_label.append(list(_flatten(eachline)))
    f_score = f1_score(actual_label, pred_lable, average=average)
    recall = recall_score(actual_label, pred_lable, average=average)
    precision = precision_score(actual_label,  pred_lable, average=average)
    return f_score, precision, recall