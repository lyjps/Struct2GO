"""
behavior_analysis.py

Analyze Struct2GO model behavior on a small subset of proteins:
- Gate value vs mean pLDDT
- Attention vs residue confidence
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import dgl

# Load your dataset and label network
def load_data(test_data_path, label_network_path):
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    label_graphs, _ = dgl.load_graphs(label_network_path)
    label_network = label_graphs[0]
    return test_dataset, label_network

# Main analysis function
def run_behavior_analysis(test_dataset, model, label_network, max_examples=5, device='cuda'):
    model.eval()
    gate_values = []
    mean_plddt_values = []
    attention_values = []

    for idx in range(min(len(test_dataset), max_examples)):
        graph, labels, seq_feature = test_dataset[idx]
        seq_feature = seq_feature.unsqueeze(0)
        batched_graph = dgl.batch([graph])

        with torch.no_grad():
            logits = model(
                batched_graph.to(device),
                seq_feature.to(device),
                label_network.to(device)
            )

        # Example gating extraction
        if hasattr(model, 'gating'):
            gate = model.gating.cpu().detach().numpy().mean()
        else:
            gate = 0.5
        gate_values.append(gate)
        
        # pLDDT
        plddt = graph.ndata['plddt']
        if plddt.dim() > 1:
            plddt = plddt.squeeze()
        mean_plddt = plddt.mean().item()
        mean_plddt_values.append(mean_plddt)

        # Example attention extraction
        if hasattr(model, 'attention'):
            att = model.attention.cpu().detach().numpy().mean()
        else:
            att = np.mean(np.ones(len(plddt)))
        attention_values.append(att)

    # Plot Gate vs Mean pLDDT
    plt.figure(figsize=(6,5))
    plt.scatter(mean_plddt_values, gate_values, color='steelblue')
    plt.xlabel('Mean pLDDT')
    plt.ylabel('Gate Value (α)')
    plt.title('Gate vs Mean pLDDT')
    plt.grid(True)
    plt.show()

    # Plot Attention vs Residue Confidence
    plt.figure(figsize=(6,5))
    plt.scatter(mean_plddt_values, attention_values, color='coral')
    plt.xlabel('Mean pLDDT')
    plt.ylabel('Attention')
    plt.title('Attention vs Residue Confidence')
    plt.grid(True)
    plt.show()

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Struct2GO Model Behavior Analysis")
    parser.add_argument('--test_data', type=str, default='new_bp_test_plddt.pkl',)
    parser.add_argument('--label_network', type=str, default='bp_label_network.dgl',
                        help='DGL file of label network')
    parser.add_argument('--model', type=str, default='save_models\mymodel_bp_network_8_0.0005_0.45_0.5314782851228244.pkl',
                        help='Path to full model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for figures')
    parser.add_argument('--max_examples', type=int, default=5, help='Number of proteins to analyze')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Load dataset and model
    test_dataset, label_network = load_data(args.test_data, args.label_network)
    model = torch.load(args.model, map_location=args.device)

    # Run analysis
    run_behavior_analysis(test_dataset, model, label_network, args.max_examples, args.device)
