"""
Confidence-Stratified Analysis for Uncertainty-Aware Structure-Function Reasoning

This module transforms the training/evaluation to investigate:
1. How different confidence regimes affect function prediction
2. Whether high pLDDT is always helpful
3. How ontologies respond differently to uncertainty
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import dgl
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc
import argparse
import warnings
import datetime
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance
import importlib
import json
from collections import defaultdict

warnings.filterwarnings('ignore')
Thresholds = list(map(lambda x: round(x*0.01, 2), list(range(60, 100))))

# ============================================================================
# CRITICAL ADDITION 1: Confidence Stratification
# ============================================================================

class ConfidenceStratifier:
    """
    Stratifies proteins by mean pLDDT to analyze confidence-dependent behavior.
    
    This is the KEY to your Nature-level question:
    "How should protein structure confidence shape function inference?"
    """
    
    def __init__(self):
        self.bins = {
            'very_high': (90, 100),   # High confidence
            'high': (70, 90),         # Moderate confidence  
            'medium': (50, 70),       # Low confidence
            'low': (0, 50)            # Very low confidence
        }
    
    def extract_mean_plddt(self, graph):
        """Extract mean pLDDT from graph node features."""
        # Assuming pLDDT is in the node features
        # Adjust index based on your feature structure
        if 'plddt' in graph.ndata:
            plddt = graph.ndata['plddt']
            # Handle case where plddt might be [N, 1] instead of [N]
            if plddt.dim() > 1:
                plddt = plddt.squeeze()
            return float(plddt.mean())
        elif 'h' in graph.ndata:
            # If pLDDT is part of the feature vector
            # You may need to adjust the index
            return float(graph.ndata['h'][:, -1].mean())  # Assuming last feature
        else:
            raise ValueError("Cannot find pLDDT in graph")
    
    def stratify_dataset(self, dataset):
        """
        Split dataset into confidence bins.
        
        Returns:
            dict: {bin_name: [indices]}
        """
        stratified_indices = defaultdict(list)
        
        for idx in range(len(dataset)):
            graph, label, seq_feature = dataset[idx]
            mean_plddt = self.extract_mean_plddt(graph)
            
            # Assign to bin
            for bin_name, (low, high) in self.bins.items():
                if low <= mean_plddt < high:
                    stratified_indices[bin_name].append(idx)
                    break
        
        return dict(stratified_indices)
    
    def create_stratified_loaders(self, dataset, stratified_indices, batch_size=8):
        """Create separate dataloaders for each confidence bin."""
        loaders = {}
        
        for bin_name, indices in stratified_indices.items():
            if len(indices) > 0:
                subset = Subset(dataset, indices)
                loaders[bin_name] = GraphDataLoader(
                    dataset=subset,
                    batch_size=batch_size,
                    drop_last=False,
                    shuffle=False
                )
                print(f"Bin '{bin_name}': {len(indices)} samples")
        
        return loaders


# ============================================================================
# CRITICAL ADDITION 2: Confidence-Aware Evaluation
# ============================================================================

class ConfidenceAwareEvaluator:
    """
    Evaluates model performance across confidence strata.
    
    This provides evidence for:
    - High pLDDT ≠ always helpful
    - Low pLDDT ≠ useless  
    - The way confidence is integrated matters
    """
    
    def __init__(self, label_network, thresholds=Thresholds):
        self.label_network = label_network
        self.thresholds = thresholds
        self.results_history = defaultdict(lambda: defaultdict(list))
    
    def evaluate_bin(self, model, dataloader, bin_name, epoch, device='cuda'):
        """Evaluate model on a specific confidence bin."""
        model.eval()
        
        all_preds = []
        all_actuals = []
        all_plddt_means = []
        
        with torch.no_grad():
            for batched_graph, labels, sequence_feature in dataloader:
                # Get predictions
                logits = model(
                    batched_graph.to(device),
                    sequence_feature.to(device),
                    self.label_network.to(device)
                )
                
                # Collect results
                preds = torch.sigmoid(logits).cpu().numpy()
                actuals = labels.cpu().numpy()
                
                all_preds.append(preds)
                all_actuals.append(actuals)
                
                # Track mean pLDDT per sample
                # Get batch assignments for each node
                batch_num_nodes = batched_graph.batch_num_nodes()
                node_batch_ids = []
                for batch_id, num_nodes in enumerate(batch_num_nodes):
                    node_batch_ids.extend([batch_id] * num_nodes)
                node_batch_ids = torch.tensor(node_batch_ids)
                
                for i in range(batched_graph.batch_size):
                    node_mask = (node_batch_ids == i)
                    if 'plddt' in batched_graph.ndata:
                        plddt_vals = batched_graph.ndata['plddt'][node_mask]
                        if plddt_vals.dim() > 1:
                            plddt_vals = plddt_vals.squeeze()
                        mean_plddt = plddt_vals.mean().item()
                    else:
                        h_vals = batched_graph.ndata['h'][node_mask, -1]
                        mean_plddt = h_vals.mean().item()
                    all_plddt_means.append(mean_plddt)
        
        # Concatenate results
        pred = np.vstack(all_preds)
        actual = np.vstack(all_actuals)
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(actual.flatten(), pred.flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        aupr = cacul_aupr(actual.flatten(), pred.flatten())
        
        # Find best threshold
        best_f1 = 0
        best_metrics = None
        
        for threshold in self.thresholds:
            f_score, precision, recall = calculate_performance(
                actual, pred, self.label_network, threshold=threshold
            )
            if f_score > best_f1:
                best_f1 = f_score
                best_metrics = {
                    'threshold': threshold,
                    'f1': f_score,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc_score,
                    'aupr': aupr,
                    'mean_plddt': np.mean(all_plddt_means),
                    'std_plddt': np.std(all_plddt_means),
                    'n_samples': len(all_plddt_means)
                }
        
        # Store history
        self.results_history[bin_name][epoch] = best_metrics
        
        return best_metrics
    
    def evaluate_all_bins(self, model, stratified_loaders, epoch, device='cuda'):
        """Evaluate across all confidence bins."""
        results = {}
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} - CONFIDENCE-STRATIFIED EVALUATION")
        print(f"{'='*80}")
        
        for bin_name, loader in stratified_loaders.items():
            metrics = self.evaluate_bin(model, loader, bin_name, epoch, device)
            results[bin_name] = metrics
            
            print(f"\n{bin_name.upper()} CONFIDENCE ({metrics['mean_plddt']:.1f} ± {metrics['std_plddt']:.1f})")
            print(f"  Samples: {metrics['n_samples']}")
            print(f"  F1:      {metrics['f1']:.4f}")
            print(f"  AUPR:    {metrics['aupr']:.4f}")
            print(f"  AUC:     {metrics['auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
        
        return results
    
    def save_results(self, filepath='confidence_stratified_results.json'):
        """Save all stratified results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results_history, f, indent=2)
        print(f"\nResults saved to {filepath}")
    
    def plot_confidence_curves(self, save_path='confidence_analysis.png'):
        """
        Plot performance across confidence bins over training.
        
        This visualization is CRITICAL for Nature:
        Shows how your model adapts to uncertainty vs baselines.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics_to_plot = ['f1', 'aupr', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            for bin_name in self.results_history.keys():
                epochs = sorted(self.results_history[bin_name].keys())
                values = [self.results_history[bin_name][e][metric] for e in epochs]
                
                ax.plot(epochs, values, marker='o', label=bin_name, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} vs Confidence Bin', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence analysis plot saved to {save_path}")


# ============================================================================
# CRITICAL ADDITION 3: Ontology-Specific Analysis
# ============================================================================

class OntologySpecificAnalyzer:
    """
    Analyzes how different GO ontologies respond to confidence.
    
    Key hypothesis: MF (molecular function) may be more structure-dependent
    than BP (biological process), so confidence affects them differently.
    """
    
    def __init__(self, go_ontology_mapping):
        """
        Args:
            go_ontology_mapping: dict mapping GO term indices to ontology type
                                 {'MF': [indices], 'BP': [indices], 'CC': [indices]}
        """
        self.ontology_mapping = go_ontology_mapping
    
    def evaluate_ontology_specific(self, actual, pred, ontology='MF'):
        """Evaluate performance on specific ontology."""
        indices = self.ontology_mapping.get(ontology, [])
        
        if len(indices) == 0:
            return None
        
        # Subset to ontology-specific terms
        actual_subset = actual[:, indices]
        pred_subset = pred[:, indices]
        
        # Calculate metrics
        aupr = cacul_aupr(actual_subset.flatten(), pred_subset.flatten())
        
        fpr, tpr, _ = roc_curve(actual_subset.flatten(), pred_subset.flatten())
        auc_score = auc(fpr, tpr)
        
        return {
            'aupr': aupr,
            'auc': auc_score,
            'n_terms': len(indices)
        }


# ============================================================================
# MODIFIED TRAINING LOOP
# ============================================================================

def train_with_confidence_analysis(args):
    """
    Main training function with confidence-stratified analysis.
    
    This is the transformation from architecture to scientific investigation.
    """
    
    # Import network
    network_file = importlib.import_module(f"model.{args.network_file}")
    SAGNetworkHierarchical = network_file.SAGNetworkHierarchical
    
    # Load datasets
    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                return dgl.DGLGraph
            return super().find_class(module, name)
    
    with open(args.train_data, 'rb') as f:
        train_dataset = DGLSafeUnpickler(f).load()
    
    with open(args.valid_data, 'rb') as f:
        valid_dataset = DGLSafeUnpickler(f).load()
    
    with open(args.label_network, 'rb') as f:
        label_network, _ = dgl.load_graphs(args.label_network)
        label_network = label_network[0]
    
    # ========================================================================
    # KEY STEP 1: Stratify validation set by confidence
    # ========================================================================
    print("\n" + "="*80)
    print("STRATIFYING VALIDATION SET BY CONFIDENCE")
    print("="*80)
    
    stratifier = ConfidenceStratifier()
    valid_stratified_indices = stratifier.stratify_dataset(valid_dataset)
    valid_stratified_loaders = stratifier.create_stratified_loaders(
        valid_dataset, valid_stratified_indices, batch_size=args.batch_size
    )
    
    # ========================================================================
    # KEY STEP 2: Initialize confidence-aware evaluator
    # ========================================================================
    evaluator = ConfidenceAwareEvaluator(label_network)
    
    # Regular training setup
    train_loader = GraphDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True
    )
    
    model = SAGNetworkHierarchical(
        56, 512, args.labels_num,
        num_convs=3,
        pool_ratio=0.75,
        dropout=args.dropout
    ).to('cuda')
    
    optimizer = optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=0.001)
    loss_fcn = nn.CrossEntropyLoss()
    
    best_overall_aupr = 0
    best_stratified_results = None
    
    # ========================================================================
    # TRAINING LOOP WITH STRATIFIED EVALUATION
    # ========================================================================
    for epoch in range(args.epochs):
        # Regular training
        model.train()
        epoch_loss = 0
        batch_num = 0
        
        for batched_graph, labels, sequence_feature in train_loader:
            logits = model(
                batched_graph.to('cuda'),
                sequence_feature.to('cuda'),
                label_network.to('cuda')
            )
            # Handle labels: [batch, 1, labels] -> [batch, labels]
            if labels.dim() == 3 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            labels = labels.float().to('cuda')
            
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_num += 1
        
        avg_loss = epoch_loss / batch_num
        print(f"\nEpoch {epoch+1}/{args.epochs} - Training Loss: {avg_loss:.4f}")
        
        # ====================================================================
        # KEY STEP 3: Confidence-stratified evaluation
        # ====================================================================
        stratified_results = evaluator.evaluate_all_bins(
            model, valid_stratified_loaders, epoch
        )
        
        # Calculate overall AUPR (for model saving)
        overall_aupr = np.mean([r['aupr'] for r in stratified_results.values()])
        
        if overall_aupr > best_overall_aupr:
            best_overall_aupr = overall_aupr
            best_stratified_results = stratified_results
            
            # Save model
            save_path = f'save_models/confidence_aware_{args.branch}_{args.network_file}_aupr{overall_aupr:.4f}.pkl'
            torch.save(model, save_path)
            print(f"\n✓ Best model saved: {save_path}")
    
    # ========================================================================
    # FINAL ANALYSIS AND VISUALIZATION
    # ========================================================================
    evaluator.save_results(f'results/{args.branch}_confidence_stratified.json')
    evaluator.plot_confidence_curves(f'results/{args.branch}_confidence_curves.png')
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL CONFIDENCE-STRATIFIED RESULTS")
    print("="*80)
    for bin_name, metrics in best_stratified_results.items():
        print(f"\n{bin_name.upper()}:")
        print(f"  AUPR: {metrics['aupr']:.4f}")
        print(f"  F1:   {metrics['f1']:.4f}")
    
    return model, evaluator, best_stratified_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--network_file', type=str, default='network')
    parser.add_argument('--learningrate', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.45)
    parser.add_argument('--train_data', type=str, default="new_bp_train_plddt.pkl")
    parser.add_argument('--valid_data', type=str, default="new_bp_valid_plddt.pkl")
    parser.add_argument('--branch', type=str, default='bp')
    parser.add_argument('--labels_num', type=int, default=809)
    parser.add_argument('--label_network', type=str, default="bp_label_network.dgl")
    parser.add_argument('--epochs', type=int, default=40)
    
    args = parser.parse_args()
    
    # Run confidence-aware training
    model, evaluator, results = train_with_confidence_analysis(args)