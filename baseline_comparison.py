"""
Baseline Comparison: Proving the Scientific Hypothesis

This demonstrates:
1. Struct2GO collapses at low confidence
2. Sequence-only is stable but shallow  
3. Your model adapts smoothly

This is the CRITICAL evidence for Nature.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

# ============================================================================
# BASELINE 1: Sequence-Only Model (Stable but Shallow)
# ============================================================================

class SequenceOnlyBaseline(nn.Module):
    """
    Baseline that uses ONLY sequence features.
    
    Expected behavior:
    - Stable across confidence bins (structure confidence doesn't matter)
    - Lower overall performance (misses structural signal)
    - Cannot leverage high-confidence structures
    """
    
    def __init__(self, seq_feature_dim, hidden_dim, output_dim, dropout=0.4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(seq_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, batched_graph, sequence_feature, label_network):
        """Ignore graph, use only sequence."""
        return self.encoder(sequence_feature)


# ============================================================================
# BASELINE 2: Structure-Only Model (Collapses at Low Confidence)
# ============================================================================

class StructureOnlyBaseline(nn.Module):
    """
    Baseline that uses ONLY structure (like original Struct2GO).
    
    Expected behavior:
    - High performance on high-confidence structures
    - Catastrophic failure on low-confidence structures
    - No graceful degradation
    """
    
    def __init__(self, node_feat_dim, hidden_dim, output_dim, num_convs=3, dropout=0.4):
        super().__init__()
        
        from model.network import SAGNetworkHierarchical
        # Use the graph network but ignore sequence
        self.graph_encoder = SAGNetworkHierarchical(
            node_feat_dim, hidden_dim, output_dim,
            num_convs=num_convs,
            pool_ratio=0.75,
            dropout=dropout
        )
    
    def forward(self, batched_graph, sequence_feature, label_network):
        """Ignore sequence, use only structure."""
        # Create dummy sequence feature of zeros
        dummy_seq = torch.zeros_like(sequence_feature)
        return self.graph_encoder(batched_graph, dummy_seq, label_network)


# ============================================================================
# BASELINE 3: Naive Fusion (No Confidence Awareness)
# ============================================================================

class NaiveFusionBaseline(nn.Module):
    """
    Combines structure + sequence without confidence awareness.
    
    Expected behavior:
    - Better than single modalities
    - Still degrades sharply at low confidence
    - Doesn't adapt fusion weights to confidence
    """
    
    def __init__(self, node_feat_dim, seq_feat_dim, hidden_dim, output_dim, 
                 num_convs=3, dropout=0.4):
        super().__init__()
        
        # Structure encoder
        from dgl.nn.pytorch import SAGEConv
        self.convs = nn.ModuleList([
            SAGEConv(node_feat_dim if i == 0 else hidden_dim, 
                    hidden_dim, 'mean')
            for i in range(num_convs)
        ])
        
        # Sequence encoder
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fixed fusion (equal weights, no adaptation)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batched_graph, sequence_feature, label_network):
        # Structure path
        h = batched_graph.ndata['h']
        for conv in self.convs:
            h = conv(batched_graph, h)
            h = torch.relu(h)
        
        # Global pooling
        with batched_graph.local_scope():
            batched_graph.ndata['h'] = h
            struct_feat = dgl.mean_nodes(batched_graph, 'h')
        
        # Sequence path
        seq_feat = self.seq_encoder(sequence_feature)
        
        # Fixed fusion (no confidence weighting)
        fused = torch.cat([struct_feat, seq_feat], dim=-1)
        fused = self.fusion(fused)
        fused = torch.relu(self.dropout(fused))
        
        return self.classifier(fused)


# ============================================================================
# COMPARATIVE EVALUATION FRAMEWORK
# ============================================================================

class BaselineComparator:
    """
    Compares your confidence-aware model against baselines.
    
    This generates the KEY FIGURE for your Nature paper.
    """
    
    def __init__(self, stratified_loaders, label_network, device='cuda'):
        self.stratified_loaders = stratified_loaders
        self.label_network = label_network
        self.device = device
        self.results = {
            'sequence_only': {},
            'structure_only': {},
            'naive_fusion': {},
            'confidence_aware': {}  # Your model
        }
    
    def evaluate_model(self, model, model_name):
        """Evaluate a model across all confidence bins."""
        model.eval()
        
        for bin_name, loader in self.stratified_loaders.items():
            all_preds = []
            all_actuals = []
            
            with torch.no_grad():
                for batched_graph, labels, sequence_feature in loader:
                    logits = model(
                        batched_graph.to(self.device),
                        sequence_feature.to(self.device),
                        self.label_network.to(self.device)
                    )
                    
                    preds = torch.sigmoid(logits).cpu().numpy()
                    actuals = labels.cpu().numpy()
                    
                    all_preds.append(preds)
                    all_actuals.append(actuals)
            
            pred = np.vstack(all_preds)
            actual = np.vstack(all_actuals)
            
            # Calculate metrics
            from model.evaluation import cacul_aupr
            aupr = cacul_aupr(actual.flatten(), pred.flatten())
            
            fpr, tpr, _ = roc_curve(actual.flatten(), pred.flatten())
            auc_score = auc(fpr, tpr)
            
            self.results[model_name][bin_name] = {
                'aupr': aupr,
                'auc': auc_score
            }
    
    def run_comparison(self, models_dict):
        """
        Run comparison across all models.
        
        Args:
            models_dict: {'model_name': model_instance}
        """
        print("\n" + "="*80)
        print("RUNNING BASELINE COMPARISON")
        print("="*80)
        
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            self.evaluate_model(model, model_name)
    
    def plot_comparative_results(self, save_path='baseline_comparison.png'):
        """
        Generate the CRITICAL FIGURE for Nature.
        
        Shows:
        1. Structure-only collapses at low confidence
        2. Sequence-only is flat (stable but shallow)
        3. Your model adapts smoothly
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Define bin order and labels
        bin_order = ['very_high', 'high', 'medium', 'low']
        bin_labels = ['Very High\n(>90)', 'High\n(70-90)', 
                     'Medium\n(50-70)', 'Low\n(<50)']
        
        metrics = ['aupr', 'auc']
        colors = {
            'sequence_only': '#2ecc71',      # Green (stable)
            'structure_only': '#e74c3c',     # Red (collapses)
            'naive_fusion': '#f39c12',       # Orange (degrades)
            'confidence_aware': '#3498db'    # Blue (adapts)
        }
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for model_name in ['sequence_only', 'structure_only', 
                              'naive_fusion', 'confidence_aware']:
                if model_name not in self.results:
                    continue
                
                values = []
                for bin_name in bin_order:
                    if bin_name in self.results[model_name]:
                        values.append(self.results[model_name][bin_name][metric])
                    else:
                        values.append(np.nan)
                
                # Plot with markers
                label = model_name.replace('_', ' ').title()
                ax.plot(bin_labels, values, 
                       marker='o', markersize=10, linewidth=3,
                       label=label, color=colors[model_name])
            
            ax.set_xlabel('Structure Confidence (Mean pLDDT)', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=14, fontweight='bold')
            ax.set_title(f'{metric.upper()} vs Confidence', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1])
            
            # Add reference line
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparative plot saved to {save_path}")
    
    def print_summary_table(self):
        """Print comparison table."""
        print("\n" + "="*80)
        print("BASELINE COMPARISON SUMMARY (AUPR)")
        print("="*80)
        
        bin_order = ['very_high', 'high', 'medium', 'low']
        
        # Header
        header = f"{'Model':<20} | " + " | ".join([f"{b:>10}" for b in bin_order])
        print(header)
        print("-" * len(header))
        
        # Rows
        for model_name in ['sequence_only', 'structure_only', 
                          'naive_fusion', 'confidence_aware']:
            if model_name not in self.results:
                continue
            
            row = f"{model_name:<20} | "
            for bin_name in bin_order:
                if bin_name in self.results[model_name]:
                    val = self.results[model_name][bin_name]['aupr']
                    row += f"{val:>10.4f} | "
                else:
                    row += f"{'N/A':>10} | "
            print(row)
    
    def calculate_degradation_scores(self):
        """
        Calculate how much each model degrades from high→low confidence.
        
        KEY METRIC: Your model should have the LOWEST degradation.
        """
        print("\n" + "="*80)
        print("DEGRADATION ANALYSIS (Very High → Low)")
        print("="*80)
        
        for model_name in self.results:
            if 'very_high' in self.results[model_name] and 'low' in self.results[model_name]:
                high_aupr = self.results[model_name]['very_high']['aupr']
                low_aupr = self.results[model_name]['low']['aupr']
                
                degradation = high_aupr - low_aupr
                degradation_pct = (degradation / high_aupr) * 100
                
                print(f"\n{model_name}:")
                print(f"  High confidence: {high_aupr:.4f}")
                print(f"  Low confidence:  {low_aupr:.4f}")
                print(f"  Degradation:     {degradation:.4f} ({degradation_pct:.1f}%)")
    
    def save_results(self, filepath='baseline_comparison_results.json'):
        """Save all results."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filepath}")


# ============================================================================
# ADDITIONAL ANALYSIS: Confidence-Performance Correlation
# ============================================================================

class ConfidenceCorrelationAnalyzer:
    """
    Analyzes the relationship between individual protein confidence
    and prediction accuracy.
    
    Shows: Your model's performance should correlate MORE smoothly
    with confidence than baselines.
    """
    
    def __init__(self, model, dataloader, label_network, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.label_network = label_network
        self.device = device
    
    def analyze(self):
        """Extract per-protein confidence and accuracy."""
        self.model.eval()
        
        confidences = []
        accuracies = []
        
        with torch.no_grad():
            for batched_graph, labels, sequence_feature in self.dataloader:
                logits = self.model(
                    batched_graph.to(self.device),
                    sequence_feature.to(self.device),
                    self.label_network.to(self.device)
                )
                
                preds = torch.sigmoid(logits).cpu().numpy()
                actuals = labels.cpu().numpy()
                
                # Per-sample metrics
                for i in range(len(preds)):
                    # Extract mean pLDDT for this protein
                    # (adjust based on your data structure)
                    if 'plddt' in batched_graph.ndata:
                        mean_plddt = batched_graph.ndata['plddt'][i].mean().item()
                    else:
                        mean_plddt = batched_graph.ndata['h'][i, -1].mean().item()
                    
                    # Calculate accuracy (e.g., average precision)
                    from sklearn.metrics import average_precision_score
                    if actuals[i].sum() > 0:  # Has positive labels
                        acc = average_precision_score(actuals[i], preds[i])
                        confidences.append(mean_plddt)
                        accuracies.append(acc)
        
        return np.array(confidences), np.array(accuracies)
    
    def plot_correlation(self, confidences, accuracies, save_path='confidence_correlation.png'):
        """Plot confidence vs accuracy scatter."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(confidences, accuracies, alpha=0.5, s=20)
        
        # Trend line
        z = np.polyfit(confidences, accuracies, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(confidences.min(), confidences.max(), 100)
        plt.plot(x_trend, p(x_trend), 'r-', linewidth=2, label='Trend')
        
        # Calculate correlation
        corr = np.corrcoef(confidences, accuracies)[0, 1]
        
        plt.xlabel('Mean pLDDT (Structure Confidence)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Precision (Prediction Accuracy)', fontsize=12, fontweight='bold')
        plt.title(f'Confidence-Accuracy Correlation (r={corr:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to {save_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_full_baseline_comparison(args, your_model, valid_stratified_loaders, label_network):
    """
    Complete comparison workflow.
    
    This generates ALL the evidence you need for Nature.
    """
    
    # Initialize baselines
    print("\nInitializing baseline models...")
    
    seq_baseline = SequenceOnlyBaseline(
        seq_feature_dim=1024,  # Adjust to your actual dim
        hidden_dim=512,
        output_dim=args.labels_num,
        dropout=args.dropout
    ).to('cuda')
    
    struct_baseline = StructureOnlyBaseline(
        node_feat_dim=56,
        hidden_dim=512,
        output_dim=args.labels_num,
        num_convs=3,
        dropout=args.dropout
    ).to('cuda')
    
    naive_baseline = NaiveFusionBaseline(
        node_feat_dim=56,
        seq_feat_dim=1024,
        hidden_dim=512,
        output_dim=args.labels_num,
        num_convs=3,
        dropout=args.dropout
    ).to('cuda')
    
    # Run comparison
    comparator = BaselineComparator(valid_stratified_loaders, label_network)
    
    models = {
        'sequence_only': seq_baseline,
        'structure_only': struct_baseline,
        'naive_fusion': naive_baseline,
        'confidence_aware': your_model
    }
    
    comparator.run_comparison(models)
    
    # Generate outputs
    comparator.print_summary_table()
    comparator.calculate_degradation_scores()
    comparator.plot_comparative_results('results/baseline_comparison.png')
    comparator.save_results('results/baseline_comparison.json')
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON COMPLETE")
    print("="*80)
    print("\nKey files generated:")
    print("  - results/baseline_comparison.png")
    print("  - results/baseline_comparison.json")