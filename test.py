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
import dgl
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import requests
from collections import defaultdict
from torch.utils.data import Subset

warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x:round(x*0.01,2), list(range(1,100))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_data', '--test_data',type=str,default='new_bp_test_plddt.pkl')
    parser.add_argument('-branch', '--branch',type=str,default='bp')
    parser.add_argument('-model','--model',type=str,default='save_models\mymodel_bp_network_b_8_0.0005_0.45_0.5307258317818497.pkl')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=809)
    parser.add_argument('-label_network', '--label_network', type=str, default='bp_label_network.dgl')
    args = parser.parse_args()
    labels_num = args.labels_num

    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                return dgl.DGLGraph
            return super().find_class(module, name) 

    with open(args.test_data,'rb')as f:
        test_dataset = DGLSafeUnpickler(f).load()
    with open(args.label_network,'rb')as f:
        label_network, _ = dgl.load_graphs(args.label_network)
        label_network = label_network[0]
    model = torch.load(args.model)

    print(f"Total test samples: {len(test_dataset)}")
    
    # ========================================================================
    # NEW: BIN PROTEINS BY pLDDT
    # ========================================================================
    bins = {
        'very_high': (90, 100),
        'high': (70, 90),
        'medium': (50, 70),
        'low': (0, 50)
    }
    
    stratified_indices = defaultdict(list)
    
    print("\nBinning proteins by confidence...")
    for idx in range(len(test_dataset)):
        graph, _, _ = test_dataset[idx]
        
        # Extract pLDDT
        plddt = graph.ndata['plddt']
        if plddt.dim() > 1:
            plddt = plddt.squeeze()
        mean_plddt = plddt.mean().item()
        
        # Assign to bin
        for bin_name, (low, high) in bins.items():
            if low <= mean_plddt < high:
                stratified_indices[bin_name].append(idx)
                break
    
    print("\nBin distribution:")
    for bin_name in ['very_high', 'high', 'medium', 'low']:
        count = len(stratified_indices[bin_name])
        pct = 100 * count / len(test_dataset)
        print(f"  {bin_name:12s}: {count:4d} samples ({pct:5.1f}%)")
    
    # ========================================================================
    # EVALUATE EACH BIN SEPARATELY
    # ========================================================================
    print("\n" + "="*80)
    print("BINNED EVALUATION RESULTS")
    print("="*80)
    
    model.eval()
    bin_results = {}
    
    for bin_name in ['very_high', 'high', 'medium', 'low']:
        indices = stratified_indices[bin_name]
        if len(indices) == 0:
            print(f"\n{bin_name.upper()}: No samples")
            continue
        
        # Create subset for this bin
        bin_subset = Subset(test_dataset, indices)
        bin_dataloader = GraphDataLoader(dataset=bin_subset, batch_size=8, drop_last=False, shuffle=False)
        
        # Evaluate this bin
        pred = []
        actual = []
        
        for batched_graph, labels, sequence_feature in bin_dataloader:
            logits = model(batched_graph.to('cuda'), sequence_feature.to('cuda'), label_network.to('cuda'))
            
            # Handle label shape
            if labels.dim() == 3 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            labels = labels.float()
            
            pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            actual.append(labels.detach().cpu().numpy())
        
        actual = np.concatenate(actual, axis=0)
        pred = np.concatenate(pred, axis=0)
        
        # Calculate metrics
        fpr, tpr, th = roc_curve(actual.flatten(), pred.flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        aupr = cacul_aupr(actual.flatten(), pred.flatten())
        
        # Find best F1
        best_f1 = 0
        best_metrics = None
        
        for threshold in Thresholds:
            f_score, precision, recall = calculate_performance(actual, pred, label_network, threshold=threshold)
            if f_score >= best_f1:
                best_f1 = f_score
                best_metrics = {
                    'threshold': threshold,
                    'f1': f_score,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc_score,
                    'aupr': aupr,
                    'n_samples': len(indices)
                }
        
        bin_results[bin_name] = best_metrics
        
        # Print results
        print(f"\n{bin_name.upper()} ({len(indices)} samples)")
        print(f"  Threshold:  {best_metrics['threshold']}")
        print(f"  F1:         {best_metrics['f1']:.4f}")
        print(f"  Precision:  {best_metrics['precision']:.4f}")
        print(f"  Recall:     {best_metrics['recall']:.4f}")
        print(f"  AUC:        {best_metrics['auc']:.4f}")
        print(f"  AUPR:       {best_metrics['aupr']:.4f}")
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: Performance vs Confidence")
    print("="*80)
    print(f"{'Bin':<12} {'N':<6} {'F1':<8} {'AUPR':<8} {'AUC':<8} {'Precision':<10} {'Recall':<8}")
    print("-"*80)
    
    for bin_name in ['very_high', 'high', 'medium', 'low']:
        if bin_name in bin_results:
            m = bin_results[bin_name]
            print(f"{bin_name:<12} {m['n_samples']:<6} {m['f1']:<8.4f} {m['aupr']:<8.4f} {m['auc']:<8.4f} {m['precision']:<10.4f} {m['recall']:<8.4f}")
    
    # Calculate degradation
    if 'very_high' in bin_results and 'low' in bin_results:
        high_aupr = bin_results['very_high']['aupr']
        low_aupr = bin_results['low']['aupr']
        degradation = (high_aupr - low_aupr) / high_aupr * 100
        print(f"\nAUPR Degradation (high→low): {degradation:.1f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    save_path = "binned_test_results.txt"
    with open(save_path, "w") as f:
        f.write(f"Branch: {args.branch}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Total samples: {len(test_dataset)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("BINNED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for bin_name in ['very_high', 'high', 'medium', 'low']:
            if bin_name in bin_results:
                m = bin_results[bin_name]
                f.write(f"{bin_name.upper()} ({m['n_samples']} samples)\n")
                f.write(f"  Threshold:  {m['threshold']}\n")
                f.write(f"  F1:         {m['f1']:.4f}\n")
                f.write(f"  Precision:  {m['precision']:.4f}\n")
                f.write(f"  Recall:     {m['recall']:.4f}\n")
                f.write(f"  AUC:        {m['auc']:.4f}\n")
                f.write(f"  AUPR:       {m['aupr']:.4f}\n\n")
        
        if 'very_high' in bin_results and 'low' in bin_results:
            high_aupr = bin_results['very_high']['aupr']
            low_aupr = bin_results['low']['aupr']
            degradation = (high_aupr - low_aupr) / high_aupr * 100
            f.write(f"\nAUPR Degradation (high→low): {degradation:.1f}%\n")
    
    print(f"\n✓ Results saved to {save_path}")