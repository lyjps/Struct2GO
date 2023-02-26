import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
import warnings
import datetime


warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x:round(x*0.01,2), list(range(1,100))))

if __name__ == "__main__":
    #参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_data', '--test_data',type=str,default='/home/jiaops/lyjps/divided_data/mf_test_dataset')
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-model','--model',type=str,default='/home/jiaops/lyjps/save_models/mymodel_mf_1_0.0005_0.45.pkl')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=273)
    args = parser.parse_args()
    labels_num = args.labels_num
    with open(args.test_data,'rb')as f:
        test_dataset = pickle.load(f)
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
            logits = model(batched_graph.to('cuda'), sequence_feature.to('cuda'))
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
    aupr=cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    score_dict = {}
    each_best_fcore = 0
    #best_fscore = 0
    each_best_scores = []
    #writer.add_pr_curve('pr_curve',actual,pred,0,num_thresholds=labels_num)
    for i in range(len(Thresholds)):
        f_score,precision, recall  = calculate_performance(actual, pred, threshold=Thresholds[i])
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores        
    t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    precision, auc_score = each_best_scores[3], each_best_scores[4] 
    print('testloss:{},t:{},f_score{}, auc{}, recall{}, precision{},aupr{}'.format(
        test_loss, t, f_score, auc_score, recall, precision,aupr))                