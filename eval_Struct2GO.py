import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
import datetime
import pandas as pd
import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x:round(x*0.01,2), list(range(1,100))))

if __name__ == "__main__":
    #参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_data', '--test_data',type=str,default='/home/jiaops/lyjps/divided_data/mf_test_dataset')
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-model','--model',type=str,default='/home/jiaops/lyjps/save_models/mymodel_mf_1_0.0005_0.45.pkl')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=273)
    parser.add_argument('-label_network', '--label_network', type=str, default='/home/jiaops/lyjps/processed_data/label_mf_network ')
    args = parser.parse_args()
    labels_num = args.labels_num
    with open(args.test_data,'rb')as f:
        test_dataset = pickle.load(f)
    with open(args.label_network,'rb')as f:
        label_network=pickle.load(f)
    model = torch.load(args.model)

    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size = 1,drop_last = False, shuffle = True)
    time = datetime.datetime.now()
    print(time)
    print('#########'+args.branch+'###########')
    print('########start testing###########') 


    t_loss = 0
    test_batch_num = 0
    pred = []
    actual = []
    model.eval()   
    for batched_graph, labels,sequence_feature  in test_dataloader:
            logits = model(batched_graph.to('cuda'), sequence_feature.to('cuda'),label_network.to('cuda'))
            labels = torch.reshape(labels,(-1,labels_num))
            loss = F.cross_entropy(logits,labels.to('cuda'))
            t_loss += loss.item()
            test_batch_num += 1
            pred.append(torch.sigmoid(logits).tolist())
            actual.append(labels.tolist())
            #writer.add_pr_curve('pr_curve',labels,logits,0)
    test_loss = "{}".format(t_loss / test_batch_num)    
    #writer.add_scalar('test/loss',test_loss,epoch)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    
    auc_values = []
    actual_array = np.array(actual)
    pred_array = np.array(pred) 
    actual_array = actual_array[:,0,:]
    pred_array = pred_array[:,0,:]
    n_labels = actual_array.shape[1]
    for i in range(n_labels):
        fpr, tpr, _ = roc_curve(actual_array[:, i].flatten(), pred_array[:, i].flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        auc_values.append(auc_score)

    aupr=cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    aupr_values = []
    y_true = np.array(actual) 
    y_scores = np.array(pred)
    y_true = y_true[:,0,:]
    y_scores = y_scores[:,0,:]
    n_labels = y_true.shape[1]
    for i in range(n_labels):
        aupr1 = average_precision_score(y_true[:, i], y_scores[:, i])
        aupr_values.append((1-0.3)*aupr1 + 0.3*aupr)

    score_dict = {}
    each_best_fcore = 0
    #best_fscore = 0
    each_best_scores = []
    #writer.add_pr_curve('pr_curve',actual,pred,0,num_thresholds=labels_num)
    for i in range(len(Thresholds)):
        f_score,precision, recall  = calculate_performance(actual, pred, label_network,threshold=Thresholds[i])
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score,auc_values,aupr_values]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores        
    t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    precision, auc_score = each_best_scores[3], each_best_scores[4] 
    # auc_values, aupr_values = each_best_scores[5],each_best_scores[6]
    print('testloss:{},t:{},f_score{}, auc{}, recall{}, precision{},aupr{}'.format(
        test_loss, t, f_score, auc_score, recall, precision,aupr))  
    # print('f_score: {}'.format(f_score))
    # print('auc_values: {}'.format(auc_values))
    # print('aupr_values: {}'.format(aupr_values))   
    # df1 = pd.DataFrame(f_score)
    # df2 = pd.DataFrame(auc_values)
    # df3 = pd.DataFrame(aupr_values)
    # df1.to_excel('f_score.xlsx', index=False, engine='openpyxl')
    # df2.to_excel('auc_values.xlsx', index=False, engine='openpyxl')
    # df3.to_excel('aupr_values.xlsx', index=False, engine='openpyxl')
    
    # bins = [i/10 for i in range(11)]
    # # 设置柱状图的宽度和位置
    # width = (bins[1] - bins[0]) / 4  # 使得三个柱子在一个bin内紧密相邻，但是不同bin之间有空隙



    # # 手动计算每组数据的直方图
    # hist_data1, _ = np.histogram(f_score, bins=bins)
    # hist_data2, _ = np.histogram(auc_values, bins=bins)
    # hist_data3, _ = np.histogram(aupr_values, bins=bins)

    # # 为每组数据设置中心点位置
    # centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    # centers1 = [center - width for center in centers]
    # centers2 = centers
    # centers3 = [center + width for center in centers]


    # # 绘制三组数据的柱状图
    # plt.bar(centers1, hist_data1, width=width, alpha=0.5, label='f_score', edgecolor='black')
    # plt.bar(centers2, hist_data2, width=width, alpha=0.5, label='auc_values', edgecolor='black')
    # plt.bar(centers3, hist_data3, width=width, alpha=0.5, label='aupr_values', edgecolor='black')

    # plt.title('Distribution of BP Test Data Sets')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.xticks(bins)
    # plt.legend(loc='upper left')  # 显示图例

    # plt.savefig('histogram3.svg', format='svg')

    # plt.show()
                
