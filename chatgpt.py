
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance
import warnings
import datetime
import dgl
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import requests
import networkx as nx

warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x: round(x*0.01, 2), list(range(1, 100))))

if __name__ == "__main__":
    # -----------------------------
    # Argument parsing
    # -----------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_data', '--test_data', type=str, default='new_cc_test_plddt.pkl')
    parser.add_argument('-branch', '--branch', type=str, default='cc')
    parser.add_argument('-model', '--model', type=str, default='save_models\mymodel_cc_temp_network_8_0.0005_0.45_0.6911459473693222.pkl')
    parser.add_argument('-labels_num', '--labels_num', type=int, default=298)
    parser.add_argument('-label_network', '--label_network', type=str, default='cc_label_network.dgl')
    args = parser.parse_args()
    labels_num = args.labels_num

    # -----------------------------
    # Safe unpickling for older DGL
    # -----------------------------
    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                return dgl.DGLGraph
            return super().find_class(module, name)

    # -----------------------------
    # Function to get AlphaFold pLDDT
    # -----------------------------
    def get_alphafold_plddt(uniprot_id):
        parser = PDBParser(QUIET=True)
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url)
        plddts = []
        for line in response.text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                plddts.append(float(line[60:66]))
        return plddts

    # -----------------------------
    # Load dataset and model
    # -----------------------------
    with open(args.test_data, 'rb') as f:
        test_dataset = DGLSafeUnpickler(f).load()
    with open(args.label_network, 'rb') as f:
        label_network, _ = dgl.load_graphs(args.label_network)
        label_network = label_network[0]
        print(type(label_network))
    model = torch.load(args.model)

    g0, y0, seq0 = test_dataset[0]

    print("SEQ_FEAT SHAPE:", seq0.shape)
    print("SEQ_FEAT pLDDT min/max:", 
        seq0[:, -1].min() if seq0.ndim > 1 else seq0[-1],
        seq0[:, -1].max() if seq0.ndim > 1 else seq0[-1])

    if "plddt" in g0.ndata:
        print("GRAPH pLDDT min/max:",
            g0.ndata["plddt"].min().item(),
            g0.ndata["plddt"].max().item())
    else:
        print("No graph.ndata['plddt']")

    print("test dataset size:", len(test_dataset))
    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size=8, drop_last=False, shuffle=True)
    print(datetime.datetime.now())
    print('#########' + args.branch + '###########')
    print('########start testing###########')

    # -----------------------------
    # Model evaluation
    # -----------------------------
    t_loss = 0
    test_batch_num = 0
    pred = []
    actual = []
    model.eval()
    #criterion = torch.nn.BCEWithLogitsLoss()

    for batched_graph, labels, sequence_feature in test_dataloader:
        logits = model(batched_graph.to('cuda'), sequence_feature.to('cuda'), label_network.to('cuda'))
        labels = torch.reshape(labels,(-1,labels_num))
        loss = F.cross_entropy(logits, labels.to('cuda'))
        t_loss += loss.item()
        test_batch_num += 1
        pred.append(torch.sigmoid(logits).detach().cpu().numpy())
        actual.append(labels.detach().cpu().numpy())

    actual = np.concatenate(actual, axis=0)
    pred = np.concatenate(pred, axis=0)
    test_loss = t_loss / test_batch_num

    fpr, tpr, _ = roc_curve(actual.flatten(), pred.flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)

    print("Avg labels per protein:", actual.sum(axis=1).mean())
    print("Positive rate:", actual.mean())
    print("Zero-label proteins:", np.sum(actual.sum(axis=1) == 0))

    # -----------------------------
    # Per-label AUC & AUPR
    # -----------------------------
    n_labels = actual.shape[1]
    auc_values = []
    aupr_values = []
    aupr = cacul_aupr(actual.flatten(), pred.flatten())
    for k in range(n_labels):
        fpr, tpr, _ = roc_curve(actual[:, k], pred[:, k])
        auc_values.append(auc(fpr, tpr))
        aupr1 = average_precision_score(actual[:, k], pred[:, k])
        aupr_values.append((1-0.3)*aupr1 + 0.3*aupr)

    # -----------------------------
    # Threshold-based Fmax selection
    # -----------------------------
    each_best_fcore = 0
    each_best_scores = []
    score_dict = {}

    """
    for i in range(len(Thresholds)):
        f_score, precision, recall = calculate_performance(actual, pred, label_network, threshold=Thresholds[i])
        if f_score >= each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, auc_values, aupr_values]
            scores = [f_score, recall, precision, auc_score]
            score_dict[Thresholds[i]] = scores

    t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    precision, auc_score = each_best_scores[3], each_best_scores[4]
    print(f'testloss:{test_loss}, t:{t}, f_score:{f_score}, auc:{auc_score}, recall:{recall}, precision:{precision}, aupr:{aupr}')
    print("HEREEEE")
    """

    # -----------------------------
    # Per-protein metrics functions
    # -----------------------------
    def protein_metrics(labels, preds, threshold=0.5): #Change to old thresholds thing

        labels = labels.squeeze()
        preds = preds.squeeze()

        assert labels.shape == preds.shape, \
            f"Shape mismatch: labels {labels.shape}, preds {preds.shape}"

        bin_preds = (preds >= threshold).astype(int)

        f_score = f1_score(labels, bin_preds, average="micro")
        aupr_score = average_precision_score(labels, preds, average="micro")

        return f_score, aupr_score

    def top_incorrect_terms(labels, preds, top_k=5):
        bin_preds = (preds >= 0.5).astype(int)
        fp = np.where((bin_preds == 1) & (labels == 0))[0]
        fn = np.where((bin_preds == 0) & (labels == 1))[0]
        return fp[:top_k], fn[:top_k]

    def select_low_conf_proteins(dataset, top_n=10):
        scores = []
        for i, (graph, _, _) in enumerate(dataset):
            avg_plddt = graph.ndata["plddt"].mean().item()
            scores.append((i, avg_plddt))
        return [i for i, _ in sorted(scores, key=lambda x: x[1])[:top_n]]

    # -----------------------------
    # Analyze low-confidence proteins
    # -----------------------------
    low_conf_indices = select_low_conf_proteins(test_dataset, top_n=10)
    protein_metrics_list = []


    #for idx in low_conf_indices:
    for idx in range(0, 100):
        labels_np = np.array(test_dataset[idx][1])
        preds_np = pred[idx]
        f_score, aupr_score = protein_metrics(labels_np, preds_np)
        protein_metrics_list.append({
            'protein_index': idx,
            'avg_plddt': test_dataset[idx][0].ndata["plddt"].mean().item(),
            #'avg_plddt': np.mean(test_dataset[idx][2][:, -1]) if test_dataset[idx][2].ndim > 1 else test_dataset[idx][2][-1],
            'Fmax': f_score,
            'AUPR': aupr_score
        })

    df_metrics = pd.DataFrame(protein_metrics_list)
    df_metrics.to_csv("per_protein_metrics.csv", index=False)
    print("Per-protein metrics for low-confidence proteins:")
    print(df_metrics)

    # -----------------------------
    # Graph visualization
    # -----------------------------

    for idx in low_conf_indices[:5]:
        seq_feat = test_dataset[idx][2]
        print(idx, seq_feat[-1])
    for idx in low_conf_indices:
        graph = test_dataset[idx][0]
        seq_feat = test_dataset[idx][2]
        preds_np = pred[idx]
        try:
            #node_size = (preds_np.max() * 500) if preds_np.ndim > 1 else 50
            plddt = graph.ndata['plddt'].cpu().numpy().squeeze()
            node_size = 10 + 0.5 * plddt
            nx_graph = graph.to_networkx().to_undirected()
            node_color = graph.ndata['plddt'].cpu().numpy() if 'plddt' in graph.ndata else np.ones(graph.number_of_nodes())
            nx.draw(nx_graph, node_color=node_color, node_size=node_size, cmap=plt.cm.Reds, with_labels=False)
            avg_plddt = graph.ndata["plddt"].mean().item()
            plt.title(f"Protein {idx} Graph (Avg pLDDT={avg_plddt:.2f})")
            #plt.title(f"Protein {idx} Graph (Avg pLDDT={np.mean(seq_feat[:, -1]) if seq_feat.ndim>1 else seq_feat[-1]:.2f})")
            plt.show()
        except Exception as e:
            print(f"Could not plot protein {idx}: {e}")

    # -----------------------------
    # Highlight top incorrect GO terms
    # -----------------------------
    for idx in low_conf_indices:
        labels_np = np.array(test_dataset[idx][1])
        preds_np = pred[idx]
        fp, fn = top_incorrect_terms(labels_np, preds_np)
        print(f"Protein {idx}:")
        print(f"  False positives GO terms: {fp}")
        print(f"  False negatives GO terms: {fn}")
