"""
Ontology-Specific Confidence Analysis

KEY HYPOTHESIS: Different GO ontologies should respond differently to structure confidence
- MF (Molecular Function): Highly structure-dependent → sensitive to confidence
- BP (Biological Process): More context-dependent → less sensitive to confidence  
- CC (Cellular Component): Location-based → intermediate sensitivity

This investigation elevates your work to Nature-level science.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from collections import defaultdict

# ============================================================================
# Ontology Mapping and Analysis
# ============================================================================

class OntologyConfidenceAnalyzer:
    """
    Analyzes how different GO ontologies respond to structure confidence.
    
    This answers the question:
    "Does the value of structural information depend on the biological question?"
    """
    
    def __init__(self, go_annotations, label_network, device='cuda'):
        """
        Args:
            go_annotations: dict mapping GO term IDs to ontology
                           {'GO:0001234': 'MF', 'GO:0005678': 'BP', ...}
            label_network: DGL graph representing GO hierarchy
            device: torch device
        """
        self.go_annotations = go_annotations
        self.label_network = label_network
        self.device = device
        
        # Create index mappings
        self.ontology_indices = self._create_ontology_indices()
        
        # Results storage
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def _create_ontology_indices(self):
        """Map ontology types to label indices."""
        ontology_map = {'MF': [], 'BP': [], 'CC': []}
        
        # Assuming you have a mapping from index to GO ID
        # You'll need to adapt this to your actual data structure
        for idx, go_id in enumerate(self.go_annotations.keys()):
            ontology = self.go_annotations[go_id]
            if ontology in ontology_map:
                ontology_map[ontology].append(idx)
        
        print("\nOntology distribution:")
        for ont, indices in ontology_map.items():
            print(f"  {ont}: {len(indices)} terms")
        
        return ontology_map
    
    def evaluate_ontology_confidence(self, model, stratified_loaders, 
                                    model_name='your_model'):
        """
        Evaluate model performance for each ontology across confidence bins.
        
        This is the KEY experiment.
        """
        model.eval()
        
        print(f"\n{'='*80}")
        print(f"ONTOLOGY-SPECIFIC CONFIDENCE ANALYSIS: {model_name}")
        print(f"{'='*80}")
        
        for bin_name, loader in stratified_loaders.items():
            print(f"\nProcessing confidence bin: {bin_name}")
            
            # Collect predictions for this bin
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
            
            # Evaluate each ontology separately
            for ontology, indices in self.ontology_indices.items():
                if len(indices) == 0:
                    continue
                
                # Subset to this ontology
                ont_pred = pred[:, indices]
                ont_actual = actual[:, indices]
                
                # Calculate metrics
                from model.evaluation import cacul_aupr
                from sklearn.metrics import roc_curve, auc
                
                aupr = cacul_aupr(ont_actual.flatten(), ont_pred.flatten())
                fpr, tpr, _ = roc_curve(ont_actual.flatten(), ont_pred.flatten())
                auc_score = auc(fpr, tpr)
                
                # Store results
                self.results[model_name][bin_name][ontology] = {
                    'aupr': aupr,
                    'auc': auc_score,
                    'n_terms': len(indices)
                }
                
                print(f"  {ontology}: AUPR={aupr:.4f}, AUC={auc_score:.4f}")
    
    def plot_ontology_confidence_heatmap(self, model_name='your_model', 
                                        save_path='ontology_confidence_heatmap.png'):
        """
        Generate heatmap showing how each ontology responds to confidence.
        
        CRITICAL VISUALIZATION for Nature paper.
        """
        # Prepare data for heatmap
        bin_order = ['very_high', 'high', 'medium', 'low']
        ontologies = ['MF', 'BP', 'CC']
        
        data = []
        for ontology in ontologies:
            row = []
            for bin_name in bin_order:
                if (bin_name in self.results[model_name] and 
                    ontology in self.results[model_name][bin_name]):
                    row.append(self.results[model_name][bin_name][ontology]['aupr'])
                else:
                    row.append(np.nan)
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(data, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   xticklabels=['Very High\n(>90)', 'High\n(70-90)', 
                               'Medium\n(50-70)', 'Low\n(<50)'],
                   yticklabels=['Molecular Function', 'Biological Process', 
                               'Cellular Component'],
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'AUPR'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_xlabel('Structure Confidence (Mean pLDDT)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('GO Ontology', fontsize=12, fontweight='bold')
        ax.set_title(f'Ontology-Specific Performance vs Confidence\n({model_name})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap saved to {save_path}")
    
    def plot_ontology_degradation(self, save_path='ontology_degradation.png'):
        """
        Plot degradation curves for each ontology.
        
        Shows: MF should degrade more than BP/CC.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bin_order = ['very_high', 'high', 'medium', 'low']
        bin_labels = ['Very High\n(>90)', 'High\n(70-90)', 
                     'Medium\n(50-70)', 'Low\n(<50)']
        
        colors = {'MF': '#e74c3c', 'BP': '#3498db', 'CC': '#2ecc71'}
        markers = {'MF': 'o', 'BP': 's', 'CC': '^'}
        
        for ontology in ['MF', 'BP', 'CC']:
            values = []
            for bin_name in bin_order:
                # Average across all models for this ontology/bin
                bin_values = []
                for model_name in self.results:
                    if (bin_name in self.results[model_name] and
                        ontology in self.results[model_name][bin_name]):
                        bin_values.append(
                            self.results[model_name][bin_name][ontology]['aupr']
                        )
                
                if bin_values:
                    values.append(np.mean(bin_values))
                else:
                    values.append(np.nan)
            
            ax.plot(bin_labels, values,
                   marker=markers[ontology],
                   markersize=12,
                   linewidth=3,
                   label=ontology,
                   color=colors[ontology])
        
        ax.set_xlabel('Structure Confidence (Mean pLDDT)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('AUPR', fontsize=14, fontweight='bold')
        ax.set_title('Ontology-Specific Sensitivity to Structure Confidence',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, title='GO Ontology', title_fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Degradation plot saved to {save_path}")
    
    def calculate_ontology_sensitivity(self):
        """
        Calculate sensitivity scores for each ontology.
        
        Sensitivity = (AUPR_high - AUPR_low) / AUPR_high
        
        Higher sensitivity = more dependent on structure quality.
        """
        print(f"\n{'='*80}")
        print("ONTOLOGY SENSITIVITY TO CONFIDENCE")
        print(f"{'='*80}")
        
        for model_name in self.results:
            print(f"\nModel: {model_name}")
            
            for ontology in ['MF', 'BP', 'CC']:
                if ('very_high' in self.results[model_name] and
                    'low' in self.results[model_name] and
                    ontology in self.results[model_name]['very_high'] and
                    ontology in self.results[model_name]['low']):
                    
                    high_aupr = self.results[model_name]['very_high'][ontology]['aupr']
                    low_aupr = self.results[model_name]['low'][ontology]['aupr']
                    
                    sensitivity = (high_aupr - low_aupr) / high_aupr
                    
                    print(f"  {ontology}:")
                    print(f"    High confidence: {high_aupr:.4f}")
                    print(f"    Low confidence:  {low_aupr:.4f}")
                    print(f"    Sensitivity:     {sensitivity:.4f} ({sensitivity*100:.1f}%)")
    
    def compare_ontology_models(self, models_dict, stratified_loaders):
        """
        Compare multiple models across ontologies and confidence.
        
        Args:
            models_dict: {'model_name': model_instance}
            stratified_loaders: confidence-stratified data loaders
        """
        for model_name, model in models_dict.items():
            self.evaluate_ontology_confidence(
                model, stratified_loaders, model_name
            )
        
        # Generate comparison visualizations
        self.plot_comparative_ontology_performance()
    
    def plot_comparative_ontology_performance(self, 
                                             save_path='ontology_model_comparison.png'):
        """
        Compare how different models handle each ontology.
        
        3x4 grid: [MF, BP, CC] x [very_high, high, medium, low]
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        bin_order = ['very_high', 'high', 'medium', 'low']
        ontologies = ['MF', 'BP', 'CC']
        
        for row, ontology in enumerate(ontologies):
            for col, bin_name in enumerate(bin_order):
                ax = axes[row, col]
                
                # Collect data for this ontology/bin
                model_names = []
                aupr_values = []
                
                for model_name in self.results:
                    if (bin_name in self.results[model_name] and
                        ontology in self.results[model_name][bin_name]):
                        model_names.append(model_name.replace('_', '\n'))
                        aupr_values.append(
                            self.results[model_name][bin_name][ontology]['aupr']
                        )
                
                # Bar plot
                bars = ax.bar(range(len(model_names)), aupr_values,
                             color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, fontsize=8, rotation=45, ha='right')
                ax.set_ylabel('AUPR', fontsize=10)
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Title
                if row == 0:
                    ax.set_title(bin_name.replace('_', ' ').title(), 
                               fontsize=11, fontweight='bold')
                if col == 0:
                    ax.text(-0.5, 0.5, ontology, 
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           verticalalignment='center',
                           rotation=90)
        
        plt.suptitle('Model Performance Across Ontologies and Confidence Levels',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0.03, 0, 1, 0.99])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparative plot saved to {save_path}")
    
    def save_results(self, filepath='ontology_confidence_analysis.json'):
        """Save all results."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


# ============================================================================
# Function-Level Confidence Analysis
# ============================================================================

class FunctionLevelAnalyzer:
    """
    Analyzes confidence effects at individual GO term level.
    
    Identifies:
    1. Which functions are most structure-dependent
    2. Which functions can be predicted from sequence alone
    3. Confidence thresholds for reliable prediction
    """
    
    def __init__(self, go_term_names, label_network):
        """
        Args:
            go_term_names: dict mapping indices to GO term names
            label_network: DGL graph
        """
        self.go_term_names = go_term_names
        self.label_network = label_network
        self.term_results = defaultdict(dict)
    
    def analyze_terms(self, model, stratified_loaders, top_k=50):
        """
        Find GO terms with highest confidence sensitivity.
        
        These are the "structure-critical" functions.
        """
        model.eval()
        
        # Collect per-term performance across confidence bins
        term_performance = defaultdict(lambda: defaultdict(list))
        
        for bin_name, loader in stratified_loaders.items():
            all_preds = []
            all_actuals = []
            
            with torch.no_grad():
                for batched_graph, labels, sequence_feature in loader:
                    logits = model(
                        batched_graph.to('cuda'),
                        sequence_feature.to('cuda'),
                        self.label_network.to('cuda')
                    )
                    
                    preds = torch.sigmoid(logits).cpu().numpy()
                    actuals = labels.cpu().numpy()
                    
                    all_preds.append(preds)
                    all_actuals.append(actuals)
            
            pred = np.vstack(all_preds)
            actual = np.vstack(all_actuals)
            
            # Calculate per-term AUPR
            from sklearn.metrics import average_precision_score
            
            for term_idx in range(pred.shape[1]):
                if actual[:, term_idx].sum() > 0:  # Has positive examples
                    aupr = average_precision_score(
                        actual[:, term_idx],
                        pred[:, term_idx]
                    )
                    term_performance[term_idx][bin_name] = aupr
        
        # Calculate sensitivity for each term
        sensitivities = []
        for term_idx, bin_scores in term_performance.items():
            if 'very_high' in bin_scores and 'low' in bin_scores:
                sensitivity = bin_scores['very_high'] - bin_scores['low']
                sensitivities.append((term_idx, sensitivity, bin_scores))
        
        # Sort by sensitivity
        sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Report top structure-dependent terms
        print(f"\n{'='*80}")
        print(f"TOP {top_k} STRUCTURE-DEPENDENT GO TERMS")
        print(f"{'='*80}\n")
        
        for rank, (term_idx, sensitivity, scores) in enumerate(sensitivities[:top_k], 1):
            term_name = self.go_term_names.get(term_idx, f"Term_{term_idx}")
            print(f"{rank}. {term_name}")
            print(f"   Sensitivity: {sensitivity:.4f}")
            print(f"   High conf: {scores.get('very_high', 0):.4f}, "
                  f"Low conf: {scores.get('low', 0):.4f}")
        
        return sensitivities


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def run_complete_ontology_analysis(args, model, valid_stratified_loaders, 
                                   label_network, go_annotations):
    """
    Complete ontology-specific confidence analysis.
    
    This generates the DEEPEST scientific insights for Nature.
    """
    
    print("\n" + "="*80)
    print("RUNNING COMPLETE ONTOLOGY ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = OntologyConfidenceAnalyzer(go_annotations, label_network)
    
    # Analyze your model
    analyzer.evaluate_ontology_confidence(
        model, valid_stratified_loaders, 'confidence_aware'
    )
    
    # Generate visualizations
    analyzer.plot_ontology_confidence_heatmap(
        'confidence_aware',
        'results/ontology_heatmap.png'
    )
    
    analyzer.plot_ontology_degradation('results/ontology_degradation.png')
    
    analyzer.calculate_ontology_sensitivity()
    
    analyzer.save_results('results/ontology_analysis.json')
    
    # Function-level analysis
    # You'll need to provide GO term names
    function_analyzer = FunctionLevelAnalyzer(
        go_term_names={},  # Populate this
        label_network=label_network
    )
    
    top_terms = function_analyzer.analyze_terms(
        model, valid_stratified_loaders, top_k=50
    )
    
    print("\n" + "="*80)
    print("ONTOLOGY ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/ontology_heatmap.png")
    print("  - results/ontology_degradation.png")
    print("  - results/ontology_analysis.json")