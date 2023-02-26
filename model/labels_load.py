#导入依赖
from cProfile import label
import pandas as pd
import pickle 
import collections
import numpy as np
import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import os
from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.glob import AvgPooling
import argparse
from network import SAGNetworkHierarchical
from torch.utils.data import Dataset


## 序列数据，转为dataframe
df = pd.read_excel('/home/jiaops/lyjps/data/seqdemo.xlsx')
seqs=dict(zip(df['Entry'], df['Sequence']))

##标签数据，确保标签和序列大小一致
labels=dict(zip(df['Entry'], df['Gene ontology IDs']))
len(labels)==len(seqs)
for i in labels:
    if not isinstance(labels[i],float):
        temp = labels[i].split(';')
        for j in range(len(temp)):
            temp[j]=temp[j].strip(' ')
        labels[i]=temp

gos=[]
namespace=collections.defaultdict(str)
is_a=collections.defaultdict(list)
part=collections.defaultdict(list)
###根据规则来提取go term ，并依据其之间的依赖关系构建图谱
with open('/home/jiaops/lyjps/data/go.obo','r')as fin:
    for line in fin:
        if '[Typedef]' in line:
            break
        if line[:5]=='id: G':
            line=line.strip().split()
            gos.append(line[1])
        elif line[:4]=='is_a':
            line=line.strip().split()
            is_a[gos[-1]].append(line[1])
        elif line[:4]=='rela' and 'part' in line:
            line=line.strip().split()
            part[gos[-1]].append(line[2])
        elif line[:5]=='names':
            line=line.strip().split()
            namespace[gos[-1]]=line[1]

for i in part:
    is_a[i].extend(part[i])

###true_path_rule
def progate(l):
    while True:
        length=len(l)
        temp=[]
        for i in l:
            temp.extend(is_a[i])
        l.update(temp)
        if len(l)==length:
            return l
        
##划分子空间，每个子空间是一个集合
bp,mf,cc=set(),set(),set()
for i in namespace:
    if namespace[i]=='biological_process':
        bp.add(i)
    elif namespace[i]=='molecular_function':
        mf.add(i)
    elif namespace[i]=='cellular_component':
        cc.add(i)

labels_with_go={}
for i in labels:
    if not isinstance(labels[i],float):
        labels_with_go[i]=progate(set(labels[i]))
len(labels),len(labels_with_go)### some items has no label are discarded

fre_counter = collections.Counter()
for i in labels_with_go:
    fre_counter.update(labels_with_go[i])
    
#label_bp,label_cc,label_mf=collections.defaultdict(list),collections.defaultdict(list),\
#collections.defaultdict(list)
#for i in labels_with_go:
#    for j in labels_with_go[i]:
#        if j in bp:
#            label_bp[i].append(j)
#        elif j in cc:
#            label_cc[i].append(j)
#        elif j in mf:
#            label_mf[i].append(j)
#print(len(label_bp))
#print(len(label_cc))
#print(len(label_mf))

df=pd.read_csv("/home/jiaops/lyjps/data/protein_list.csv",sep=" ")
list1=df.values.tolist()
protein_list = np.array(list1)
print(protein_list.shape)
label_bp=collections.defaultdict(list)
label_mf=collections.defaultdict(list)
label_cc=collections.defaultdict(list)
protein_list_final = []
for i in labels_with_go:
    if i in protein_list:
        protein_list_final.append(i)
        for j in labels_with_go[i]:
            if j in bp:
                label_bp[i].append(j)
            elif j in mf:
                label_mf[i].append(j)
            elif j in cc:
                label_cc[i].append(j)         
                #temp = []
                #temp.append(i)
                #temp.append(j)
                #label_bp.append(temp)
                    

#print(len(label_bp))
#print(label_bp)
bp_c=collections.Counter()


for i in label_bp:
    #for j in label_bp[i]:
    bp_c.update(label_bp[i])

bp_d=dict(bp_c)
bp_set=set()
for i in bp_d:
    if bp_d[i]>=250:
        bp_set.add(i)
       
print(len(bp_set))

cc_c=collections.Counter()
for i in label_cc:
    cc_c.update(label_cc[i])
cc_d=dict(cc_c)
cc_set=set()
for i in cc_d:
    if cc_d[i]>=100:
        cc_set.add(i)

print(len(cc_set))        


mf_c=collections.Counter()
for i in label_mf:
    mf_c.update(label_mf[i])
mf_d=dict(mf_c)
mf_set=set()
for i in mf_d:
    if mf_d[i]>=100:
        mf_set.add(i)

print(len(mf_set))

def goterm2idx(term_set):
    term_dict=dict(enumerate(term_set))
    term_dict={v:k for k,v in term_dict.items()}
    return term_dict

bp_term2idx=goterm2idx(bp_set)
mf_term2idx=goterm2idx(mf_set)
cc_term2idx=goterm2idx(cc_set)


def labels2onehot(labels,index):
    labels_new={}
    l=len(index)
    for i in labels:
        temp = [0]*l
        for j in labels[i]:
            if(j in bp_set or j in mf_set or j in cc_set):
                temp[index[j]]=1
        labels_new[i]=temp
    return labels_new

bp_label2onehot=labels2onehot(label_bp,bp_term2idx)
bp_entry=list(label_bp.keys())

mf_label2onehot=labels2onehot(label_mf,mf_term2idx)
mf_entry=list(label_mf.keys())

cc_label2onehot=labels2onehot(label_cc,cc_term2idx)
cc_entry=list(label_cc.keys())



graph_dic_bp = {}
graph_dic_mf = {}
graph_dic_cc = {}
for path,dir_list,file_list in os.walk("/home/jiaops/lyjps/data/proteins_edgs"):  
    for file_name in file_list: 
        trace = os.path.join(path, file_name)
        name = file_name.split(".")[0]
        if trace.endswith(".txt"):
            if(name in bp_entry):
                graph_dic_bp[name] = pd.read_csv(trace, names=['Src','Dst'],header=None, sep=" ")
            if(name in mf_entry):
                graph_dic_mf[name] = pd.read_csv(trace, names=['Src','Dst'],header=None, sep=" ")
            if(name in cc_entry):
                graph_dic_cc[name] = pd.read_csv(trace, names=['Src','Dst'],header=None, sep=" ")        

feature_dic_bp = {}
feature_dic_mf = {}
feature_dic_cc = {}
with open('/home/jiaops/lyjps/processed_data/protein_node2onehot','rb')as f:
    protein_node2onehot = pickle.load(f) 
print(len(protein_node2onehot))

for path,dir_list,file_list in os.walk("/home/jiaops/lyjps/data/struct_feature"):  
    for file_name in file_list: 
        trace = os.path.join(path, file_name)
        name = file_name.split(".")[0]
        if trace.endswith(".csv"):
            if(name in bp_entry):
                #feature_dic_bp[name] = np.loadtxt(trace, delimiter = ' ')
                #feature_dic_bp[name] =  np.hstack((protein_node2onehot[name],np.loadtxt(trace, delimiter = ' ')))
                feature_dic_bp[name] = protein_node2onehot[name]
            if(name in mf_entry):
                #feature_dic_mf[name] = np.loadtxt(trace, delimiter = ' ')
                #feature_dic_mf[name] =  np.hstack((protein_node2onehot[name],np.loadtxt(trace, delimiter = ' ')))
                feature_dic_mf[name] = protein_node2onehot[name]
            if(name in cc_entry):
                #feature_dic_cc[name] = np.loadtxt(trace, delimiter = ' ')   
                #feature_dic_cc[name] =  np.hstack((protein_node2onehot[name],np.loadtxt(trace, delimiter = ' ')))  
                feature_dic_cc[name] = protein_node2onehot[name]



with open('/home/jiaops/lyjps/processed_data/dict_sequence_feature','rb')as f:
    dict_sequence_feature = pickle.load(f)

emb_graph_mf = {}
emb_seq_feature_mf = {}
emb_label_mf = {}

for i in mf_entry:
    edges_data = graph_dic_mf[i]
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    nodes_feature = feature_dic_mf[i]
    g.ndata['feature'] = torch.tensor(nodes_feature, dtype=torch.float32)
    labels = mf_label2onehot[i]
    labels = np.array(labels).reshape(1,273)
    labels = torch.tensor(labels.astype(np.float32))
    emb_graph_mf[i] = g
    emb_seq_feature_mf[i] = torch.tensor(dict_sequence_feature[i].astype(np.float32))
    emb_label_mf[i] = labels

with open('/home/jiaops/lyjps/processed_data/emb_graph_mf_without_Node2vec ','wb')as f:
    pickle.dump(emb_graph_mf,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_seq_feature_mf ','wb')as f:
#     pickle.dump(emb_seq_feature_mf,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_label_mf ','wb')as f:
#     pickle.dump(emb_label_mf,f)    


emb_graph_cc = {}
emb_seq_feature_cc = {}
emb_label_cc = {}

for i in cc_entry:
    edges_data = graph_dic_cc[i]
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    nodes_feature = feature_dic_cc[i]
    g.ndata['feature'] = torch.tensor(nodes_feature, dtype=torch.float32)
    labels = cc_label2onehot[i]
    labels = np.array(labels).reshape(1,298)
    labels = torch.tensor(labels.astype(np.float32))
    emb_graph_cc[i] = g
    emb_seq_feature_cc[i] = torch.tensor(dict_sequence_feature[i].astype(np.float32))
    emb_label_cc[i] = labels

with open('/home/jiaops/lyjps/processed_data/emb_graph_cc_without_Node2vec ','wb')as f:
    pickle.dump(emb_graph_cc,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_seq_feature_cc ','wb')as f:
#     pickle.dump(emb_seq_feature_cc,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_label_cc ','wb')as f:
#     pickle.dump(emb_label_cc,f)    

emb_graph_bp = {}
emb_seq_feature_bp = {}
emb_label_bp = {}

for i in bp_entry:
    edges_data = graph_dic_bp[i]
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    nodes_feature = feature_dic_bp[i]
    g.ndata['feature'] = torch.tensor(nodes_feature, dtype=torch.float32)
    labels = bp_label2onehot[i]
    labels = np.array(labels).reshape(1,809)
    labels = torch.tensor(labels.astype(np.float32))
    emb_graph_bp[i] = g
    emb_seq_feature_bp[i] = torch.tensor(dict_sequence_feature[i].astype(np.float32))
    emb_label_bp[i] = labels

with open('/home/jiaops/lyjps/processed_data/emb_graph_bp_without_Node2vec','wb')as f:
    pickle.dump(emb_graph_bp,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_seq_feature_bp ','wb')as f:
#     pickle.dump(emb_seq_feature_bp,f)    

# with open('/home/jiaops/lyjps/processed_data/emb_label_bp ','wb')as f:
#     pickle.dump(emb_label_bp,f)    