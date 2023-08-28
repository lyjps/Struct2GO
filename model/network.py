import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv,GATConv, AvgPooling, MaxPooling
from model.layer import ConvPoolBlock, SAGPool


class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs:int=3,
                 pool_ratio:float=0.5, dropout:float=0.5):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs
       #self.classify = torch.nn.Linear(hid_dim, out_dim)
        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))
        self.convpools = torch.nn.ModuleList(convpools)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hid_dim * 2 + 1024, nhead=8), num_layers=6)    
        self.lin1 = torch.nn.Linear(hid_dim * 2 + 1024, hid_dim * 2)
        #self.lin1 = torch.nn.Linear(hid_dim * 2, hid_dim)
        #self.lin1 = torch.nn.Linear(1024,hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim*2, hid_dim)
        self.lin3 = torch.nn.Linear(hid_dim, out_dim)
        self.label_network1 = GATConv(1,1,num_heads=8,allow_zero_in_degree=True)

        self.line_new = torch.nn.Linear(hid_dim * 2 + 1024, out_dim)



    def update_parent_features(self,label_network:dgl.DGLGraph, labels):
        # 获取图中的所有边
        edges = label_network.edges()

        second_dim_elements = labels[0,:]
        # 对于图中的每条边
        for child_idx, parent_idx in zip(edges[0], edges[1]):
            # 如果child节点的特征值大于parent节点的特征值
            if second_dim_elements[child_idx] > second_dim_elements[parent_idx]:
                # 更新parent节点的特征值为child节点的特征值
                second_dim_elements[parent_idx] = second_dim_elements[child_idx]
         # 更新labels的第二列为second_dim_elements
        labels[0, :] = second_dim_elements
        return labels
    

    def forward(self, graph:dgl.DGLGraph, sequence_feature,label_network:dgl.DGLGraph):
        feat = graph.ndata["feature"]
        final_readout = None

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        final_readout = torch.cat((final_readout,sequence_feature), -1)
        #con_readout = self.transformer_encoder(final_readout)
        #final_readout = torch.cat((sequence_feature,con_readout), -1)
        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)
        feat = feat.t()
        max_value,_ = torch.max(self.label_network1(label_network,feat),dim=1)
        feat - F.relu(max_value)
        feat = feat.t()
        # feat = self.update_parent_features(label_network, feat)
        #feat = self.line_new(final_readout)
        #feat = torch.sigmoid(feat)
        
        return feat



class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs=3,
                 pool_ratio:float=0.5, dropout:float=0.0):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2 + 1024, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)
    
    def forward(self, graph:dgl.DGLGraph, sequence_feature):
        feat = graph.ndata["feature"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
        
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat([self.avg_readout(graph, feat), self.max_readout(graph, feat)], dim=-1)
        feat = torch.cat((feat,sequence_feature),-1)
        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)

        return feat



def get_sag_network(net_type:str="hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError("SAGNetwork type {} is not supported.".format(net_type))
