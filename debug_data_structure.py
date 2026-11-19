"""
Debug script to understand your data structure.
Run this to see exactly how your data is organized.
"""

import pickle
import torch
import dgl
from dgl.dataloading import GraphDataLoader

print("="*80)
print("DATA STRUCTURE DEBUG")
print("="*80)

# Load data
class DGLSafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DGLHeteroGraph':
            return dgl.DGLGraph
        return super().find_class(module, name)

print("\n1. Loading single sample...")
with open('new_bp_valid_plddt.pkl', 'rb') as f:
    valid_dataset = DGLSafeUnpickler(f).load()

print(f"Dataset size: {len(valid_dataset)}")

# Examine single sample
graph, label, seq_feature = valid_dataset[0]

print("\n2. Single Graph Structure:")
print(f"   Type: {type(graph)}")
print(f"   Num nodes: {graph.num_nodes()}")
print(f"   Num edges: {graph.num_edges()}")
print(f"   Node data keys: {list(graph.ndata.keys())}")

for key in graph.ndata.keys():
    print(f"   - {key}: shape {graph.ndata[key].shape}, dtype {graph.ndata[key].dtype}")

print(f"\n   Label shape: {label.shape}")
print(f"   Sequence feature shape: {seq_feature.shape}")

# Check for pLDDT
print("\n3. Checking for pLDDT:")
if 'plddt' in graph.ndata:
    plddt = graph.ndata['plddt']
    print(f"   ✓ Found in ndata['plddt']")
    print(f"     Shape: {plddt.shape}")
    print(f"     Mean: {plddt.mean().item():.2f}")
    print(f"     Range: [{plddt.min().item():.2f}, {plddt.max().item():.2f}]")
    
    # If it has extra dimension, show it
    if plddt.dim() > 1:
        print(f"     Note: Has shape {plddt.shape}, might need squeeze()")
        plddt_squeezed = plddt.squeeze()
        print(f"     After squeeze: {plddt_squeezed.shape}")
        print(f"     Mean (squeezed): {plddt_squeezed.mean().item():.2f}")
elif 'h' in graph.ndata:
    h = graph.ndata['h']
    print(f"   Found in ndata['h']")
    print(f"     Full feature shape: {h.shape}")
    print(f"     Assuming pLDDT is last feature...")
    print(f"     Last feature mean: {h[:, -1].mean().item():.2f}")
    print(f"     Last feature range: [{h[:, -1].min().item():.2f}, {h[:, -1].max().item():.2f}]")
else:
    print("   ✗ Cannot find pLDDT!")

# Test batching
print("\n4. Testing Batch Behavior:")
dataloader = GraphDataLoader(dataset=valid_dataset, batch_size=8, drop_last=False, shuffle=False)

for batched_graph, labels, seq_features in dataloader:
    print(f"\n   Batch info:")
    print(f"   - batch_size: {batched_graph.batch_size}")
    print(f"   - Total nodes: {batched_graph.num_nodes()}")
    print(f"   - batch_num_nodes: {batched_graph.batch_num_nodes()}")
    print(f"   - Labels shape: {labels.shape}")
    print(f"   - Seq features shape: {seq_features.shape}")
    
    # Show how to correctly extract per-sample info
    print(f"\n   Per-sample node counts:")
    batch_num_nodes = batched_graph.batch_num_nodes()
    for i, num_nodes in enumerate(batch_num_nodes):
        print(f"     Sample {i}: {num_nodes} nodes")
    
    # Correct way to get node-to-batch mapping
    print(f"\n   Creating node-to-batch mapping:")
    node_batch_ids = []
    for batch_id, num_nodes in enumerate(batch_num_nodes):
        node_batch_ids.extend([batch_id] * num_nodes)
    node_batch_ids = torch.tensor(node_batch_ids)
    print(f"   - Total nodes mapped: {len(node_batch_ids)}")
    print(f"   - Matches graph nodes: {len(node_batch_ids) == batched_graph.num_nodes()}")
    
    # Extract pLDDT for each sample
    print(f"\n   Extracting per-sample mean pLDDT:")
    if 'plddt' in batched_graph.ndata:
        plddt_data = batched_graph.ndata['plddt']
        print(f"   pLDDT data shape: {plddt_data.shape}")
        
        for i in range(batched_graph.batch_size):
            node_mask = (node_batch_ids == i)
            sample_plddt = plddt_data[node_mask]
            
            # Handle different shapes
            if sample_plddt.dim() > 1:
                sample_plddt = sample_plddt.squeeze()
            
            mean_plddt = sample_plddt.mean().item()
            print(f"     Sample {i}: {mean_plddt:.2f} (from {node_mask.sum()} nodes)")
    
    # Only process first batch for debugging
    break

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
print("\nKey takeaways:")
print("1. Check the shapes above to understand your data")
print("2. If pLDDT has shape [N, 1], you need to squeeze it")
print("3. Use batch_num_nodes() to correctly map nodes to samples")
print("4. The master script should now work with the fix applied")