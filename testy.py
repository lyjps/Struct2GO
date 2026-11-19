"""
MODEL BEHAVIOR ANALYSIS PIPELINE
================================
Generates interpretability figures for your Nature paper:

1. Gate value vs mean pLDDT (per-protein)
2. Attention vs residue confidence (per-residue)
"""

import torch
import dgl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# MechanismExtractor from your pipeline
class MechanismExtractor:
    """
    Extracts gating weights and attention maps from your model.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        
    def register_hooks(self):
        """Register forward hooks to capture intermediate values."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Adjust these based on your model's actual layers
        if hasattr(self.model, 'gating'):
            self.model.gating.register_forward_hook(get_activation('gating'))
        
        if hasattr(self.model, 'attention'):
            self.model.attention.register_forward_hook(get_activation('attention'))
        
        for i, layer in enumerate(self.model.modules()):
            if 'GATConv' in str(type(layer)) or 'GAT' in str(type(layer)):
                layer.register_forward_hook(get_activation(f'gat_{i}'))
    
    def extract_mechanisms(self, graph, seq_feature, label_network, device='cuda'):
        """Run model and extract gating + attention."""
        self.activations = {}
        batched_graph = dgl.batch([graph])
        seq_feature = seq_feature.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                batched_graph.to(device),
                seq_feature.to(device),
                label_network.to(device)
            )
        
        mechanisms = {}
        if 'gating' in self.activations:
            gating = self.activations['gating']
            mechanisms['gating_weight'] = gating.cpu().numpy()
        
        attention_weights = []
        for key in self.activations:
            if 'gat' in key or 'attention' in key:
                attention_weights.append(self.activations[key].cpu().numpy())
        
        if attention_weights:
            mechanisms['node_attention'] = np.mean(attention_weights, axis=0)
        
        return mechanisms

# ============================================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    class DGLSafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'DGLHeteroGraph':
                import dgl
                return dgl.DGLGraph
            return super().find_class(module, name)
    
    with open(args.test_data, 'rb') as f:
        test_dataset = DGLSafeUnpickler(f).load()
    
    # Load label network
    label_network, _ = dgl.load_graphs(args.label_network)
    label_network = label_network[0]
    
    # Load model
    model = torch.load(args.model, map_location='cpu')
    extractor = MechanismExtractor(model)
    extractor.register_hooks()
    
    # Collect lists
    gate_values = []
    mean_plddts = []
    all_attentions = []
    all_plddts = []
    
    print("Extracting mechanisms across all proteins...")
    for idx in range(len(test_dataset)):
        print(f"Processing protein {idx+1}/{len(test_dataset)}", end='\r')
        graph, labels, seq_feature = test_dataset[idx]
        mechanisms = extractor.extract_mechanisms(graph, seq_feature, label_network, device='cpu')
        
        # Gate value
        gating = mechanisms.get('gating_weight', np.array([0.5]))[0]
        plddt = graph.ndata['plddt']
        if plddt.dim() > 1:
            plddt = plddt.squeeze()
        mean_plddt = plddt.mean().item()
        
        gate_values.append(gating)
        mean_plddts.append(mean_plddt)
        
        # Attention per residue
        attention = mechanisms.get('node_attention', None)
        if attention is not None:
            all_attentions.extend(attention.flatten())
            all_plddts.extend(plddt.numpy().flatten())
    
    # Plot 1: Gate vs mean pLDDT
    plt.figure(figsize=(8,6))
    plt.scatter(mean_plddts, gate_values, alpha=0.6, edgecolors='black')
    plt.xlabel('Mean pLDDT')
    plt.ylabel('Gating Weight (Structure)')
    plt.title('Gate Value vs Mean pLDDT')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_vs_plddt.png', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'gate_vs_plddt.png'}")
    
    # Plot 2: Attention vs residue pLDDT
    plt.figure(figsize=(8,6))
    plt.scatter(all_plddts, all_attentions, alpha=0.3)
    plt.xlabel('Residue pLDDT')
    plt.ylabel('Node Attention')
    plt.title('Attention vs Residue Confidence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_vs_plddt.png', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'attention_vs_plddt.png'}")
    
    print("\nModel Behavior Analysis complete!")

# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model behavior analysis plots")
    
    parser.add_argument('--test_data', type=str, default='new_bp_test_plddt.pkl',)
    parser.add_argument('--label_network', type=str, default='bp_label_network.dgl',
                        help='DGL file of label network')
    parser.add_argument('--model', type=str, default='save_models\mymodel_bp_network_8_0.0005_0.45_0.5314782851228244.pkl',
                        help='Path to full model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    main(args)
