import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Load saved data
data = torch.load("attn_plddt.pt")
all_attn = data["attn"]   # list of tensors per protein
all_plddt = data["plddt"] # list of tensors per protein

print("=" * 80)
print("COMPREHENSIVE ATTENTION-PLDDT ANALYSIS")
print("=" * 80)

# =============================================================================
# DATA PREPARATION
# =============================================================================

n_proteins = len(all_attn)
n_heads = all_attn[0].squeeze().shape[0]

# Store data for each protein and head
attn_plddt_matrix = np.zeros((n_proteins, n_heads))
correlations_per_head = [[] for _ in range(n_heads)]
all_correlations = []

for i, (attn, plddt) in enumerate(zip(all_attn, all_plddt)):
    attn_flat = attn.squeeze().cpu().numpy()  # Shape: [n_heads]
    plddt_flat = plddt.squeeze().cpu().numpy()  # Shape: [N_residues]
    
    # Ensure arrays are at least 1D
    if attn_flat.ndim == 0:
        attn_flat = np.array([attn_flat.item()])
    if plddt_flat.ndim == 0:
        plddt_flat = np.array([plddt_flat.item()])
    
    # Segment pLDDT to match attention dimensions
    segment_size = len(plddt_flat) // len(attn_flat)
    plddt_segments = []
    
    for j in range(len(attn_flat)):
        start_idx = j * segment_size
        end_idx = start_idx + segment_size if j < len(attn_flat) - 1 else len(plddt_flat)
        plddt_segments.append(np.mean(plddt_flat[start_idx:end_idx]))
    
    plddt_segments = np.array(plddt_segments)
    
    # Store attention × mean pLDDT for heatmap
    attn_plddt_matrix[i, :] = attn_flat * plddt_segments
    
    # Compute correlation per head (with safety checks)
    for j in range(n_heads):
        if len(attn_flat) > 1:
            # Check if arrays have variance (not constant)
            if np.std(attn_flat) > 1e-10 and np.std(plddt_segments) > 1e-10:
                corr, _ = stats.pearsonr(attn_flat, plddt_segments)
                if not np.isnan(corr):
                    correlations_per_head[j].append(corr)
    
    # Overall correlation for this protein (with safety checks)
    if len(attn_flat) > 1:
        if np.std(attn_flat) > 1e-10 and np.std(plddt_segments) > 1e-10:
            corr, _ = stats.pearsonr(attn_flat, plddt_segments)
            if not np.isnan(corr):
                all_correlations.append(corr)

# Convert to arrays
correlations_per_head = [np.array(corrs) if len(corrs) > 0 else np.array([0.0]) for corrs in correlations_per_head]
all_correlations = np.array(all_correlations) if len(all_correlations) > 0 else np.array([0.0])

print(f"Analyzed {n_proteins} proteins with {n_heads} attention heads each")
print(f"Valid correlations computed: {len(all_correlations)}")
if len(all_correlations) > 0:
    print(f"Overall mean correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
else:
    print("Warning: No valid correlations could be computed (constant inputs)")

# =============================================================================
# FIGURE 1: HEATMAP OF ATTENTION × MEAN PLDDT
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(10, 12))

# Sort proteins by mean attention×pLDDT for better visualization
row_means = np.mean(attn_plddt_matrix, axis=1)
sorted_indices = np.argsort(row_means)[::-1]
sorted_matrix = attn_plddt_matrix[sorted_indices, :]

im = ax1.imshow(sorted_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax1.set_xlabel('Attention Head Index', fontsize=13, fontweight='bold')
ax1.set_ylabel('Protein Index (sorted by mean value)', fontsize=13, fontweight='bold')
ax1.set_title('Attention × Mean pLDDT Across All Proteins\n(rows = proteins, columns = attention heads)', 
              fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Attention × Mean pLDDT', fontsize=12, fontweight='bold')

# Add grid for better readability
ax1.set_xticks(np.arange(n_heads))
ax1.set_xticklabels(np.arange(n_heads))
ax1.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Add summary statistics as text
textstr = f'Mean: {np.mean(sorted_matrix):.3f}\nStd: {np.std(sorted_matrix):.3f}\nMax: {np.max(sorted_matrix):.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('fig1_attention_plddt_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure 1 saved: fig1_attention_plddt_heatmap.png")

# =============================================================================
# FIGURE 2: BARPLOT OF MEAN ± STD CORRELATION PER HEAD
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 7))

# Calculate mean and std for each head
head_means = [np.mean(corrs) if len(corrs) > 0 else 0.0 for corrs in correlations_per_head]
head_stds = [np.std(corrs) if len(corrs) > 0 else 0.0 for corrs in correlations_per_head]
head_indices = np.arange(n_heads)

# Create bar plot
bars = ax2.bar(head_indices, head_means, yerr=head_stds, 
               capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5,
               color=plt.cm.RdYlGn(0.5 + np.array(head_means) * 0.5))

# Add horizontal line at zero
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Customize plot
ax2.set_xlabel('Attention Head Index', fontsize=13, fontweight='bold')
ax2.set_ylabel('Mean Correlation (± Std Dev)', fontsize=13, fontweight='bold')
ax2.set_title('Attention-pLDDT Correlation per Attention Head\n(averaged across all proteins)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(head_indices)
ax2.set_xticklabels([f'Head {i}' for i in head_indices])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (mean, std) in enumerate(zip(head_means, head_stds)):
    height = mean + std if mean > 0 else mean - std
    ax2.text(i, height + 0.02 * (1 if mean > 0 else -1), 
             f'{mean:.3f}', ha='center', va='bottom' if mean > 0 else 'top',
             fontsize=9, fontweight='bold')

# Add summary box
summary_text = f'Overall Statistics:\n'
summary_text += f'Mean corr: {np.mean(head_means):.3f}\n'
summary_text += f'Std corr: {np.mean(head_stds):.3f}\n'
summary_text += f'Range: [{np.min(head_means):.3f}, {np.max(head_means):.3f}]'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax2.text(0.98, 0.97, summary_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props, 
         family='monospace')

plt.tight_layout()
plt.savefig('fig2_correlation_per_head.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved: fig2_correlation_per_head.png")

# =============================================================================
# FIGURE 3: MODEL PERFORMANCE VS MASKING (SIMULATION)
# =============================================================================

fig3, axes = plt.subplots(2, 2, figsize=(14, 12))

# SIMULATE performance data - replace with actual model predictions
# In real analysis, you would:
# 1. Mask residues with low attention×pLDDT
# 2. Mask residues with high attention×pLDDT  
# 3. Re-run predictions and measure accuracy/loss

mask_percentages = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# Simulated data - replace with actual measurements
np.random.seed(42)
baseline_acc = 0.85

# Masking high attention×pLDDT regions should hurt performance more
mask_high_acc = baseline_acc - (mask_percentages / 100) * 0.6 + np.random.normal(0, 0.02, len(mask_percentages))
# Masking low attention×pLDDT regions should hurt less
mask_low_acc = baseline_acc - (mask_percentages / 100) * 0.3 + np.random.normal(0, 0.02, len(mask_percentages))

mask_high_loss = 0.4 + (mask_percentages / 100) * 1.2 + np.random.normal(0, 0.05, len(mask_percentages))
mask_low_loss = 0.4 + (mask_percentages / 100) * 0.6 + np.random.normal(0, 0.05, len(mask_percentages))

# Plot 1: Accuracy vs masking percentage
ax = axes[0, 0]
ax.plot(mask_percentages, mask_high_acc, 'o-', linewidth=2.5, markersize=8, 
        label='Mask High Attention×pLDDT', color='crimson')
ax.plot(mask_percentages, mask_low_acc, 's-', linewidth=2.5, markersize=8,
        label='Mask Low Attention×pLDDT', color='steelblue')
ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, 
           label=f'Baseline (no masking)', alpha=0.7)
ax.set_xlabel('Percentage of Residues Masked (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy vs. Masking Strategy', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim([0, 1])

# Plot 2: Loss vs masking percentage
ax = axes[0, 1]
ax.plot(mask_percentages, mask_high_loss, 'o-', linewidth=2.5, markersize=8,
        label='Mask High Attention×pLDDT', color='crimson')
ax.plot(mask_percentages, mask_low_loss, 's-', linewidth=2.5, markersize=8,
        label='Mask Low Attention×pLDDT', color='steelblue')
ax.set_xlabel('Percentage of Residues Masked (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Prediction Loss', fontsize=12, fontweight='bold')
ax.set_title('Model Loss vs. Masking Strategy', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)

# Plot 3: Performance drop comparison
ax = axes[1, 0]
acc_drop_high = baseline_acc - mask_high_acc
acc_drop_low = baseline_acc - mask_low_acc
x = np.arange(len(mask_percentages))
width = 0.35
ax.bar(x - width/2, acc_drop_high, width, label='Mask High', color='crimson', alpha=0.7)
ax.bar(x + width/2, acc_drop_low, width, label='Mask Low', color='steelblue', alpha=0.7)
ax.set_xlabel('Masking Percentage', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy Drop from Baseline', fontsize=12, fontweight='bold')
ax.set_title('Performance Impact by Masking Strategy', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}%' for p in mask_percentages], rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Statistical significance
ax = axes[1, 1]
differences = acc_drop_high - acc_drop_low
ax.plot(mask_percentages, differences, 'o-', linewidth=2.5, markersize=10, 
        color='darkgreen')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between(mask_percentages, 0, differences, alpha=0.3, color='green')
ax.set_xlabel('Percentage of Residues Masked (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Differential Impact\n(High mask - Low mask)', fontsize=12, fontweight='bold')
ax.set_title('Importance of High Attention×pLDDT Regions', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Add interpretation text
textstr = 'Interpretation:\nMasking high attention×pLDDT\nregions causes larger\nperformance degradation,\nsuggesting these regions are\nmore critical for predictions.'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

fig3.suptitle('Model Performance Analysis: Masking Experiments\n[NOTE: Replace with actual model predictions]',
              fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('fig3_masking_performance.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved: fig3_masking_performance.png")
print("  ⚠ WARNING: This figure uses simulated data. Replace with actual model performance metrics.")

# =============================================================================
# FIGURE 4 (OPTIONAL): 3D STRUCTURE VISUALIZATION GUIDE
# =============================================================================

fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

# Since we don't have actual 3D coordinates, create a guide figure
ax = axes[0, 0]
ax.text(0.5, 0.5, 'FIGURE 4: 3D STRUCTURE VISUALIZATION\n\n'
        'To create this figure, you need:\n\n'
        '1. PDB files or 3D coordinates for proteins\n'
        '2. PyMOL, Py3Dmol, or NGLview for visualization\n'
        '3. Map attention×pLDDT values to residue colors\n\n'
        'Recommended approach:\n'
        '• Use PyMOL: color residues by attention×pLDDT\n'
        '• Highlight top 10% residues in red\n'
        '• Highlight bottom 10% residues in blue\n'
        '• Show if high-attention regions cluster\n'
        '  near active sites or binding interfaces',
        transform=ax.transAxes, fontsize=11, verticalalignment='center',
        horizontalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='lightcyan', alpha=0.8), family='monospace')
ax.axis('off')
ax.set_title('3D Structure Visualization (Setup Guide)', fontsize=13, fontweight='bold')

# Example pseudocode
ax = axes[0, 1]
code_text = '''# Example PyMOL script
from pymol import cmd
import numpy as np

# Load structure
cmd.load("protein.pdb")

# Color by attention×pLDDT
for i, val in enumerate(attn_plddt_values):
    # Map value to color (red=high, blue=low)
    color = get_color(val)
    cmd.color(color, f"resi {i+1}")

# Highlight key regions
cmd.select("high_attn", "resi 1+5+12+...")
cmd.show("spheres", "high_attn")

cmd.save("protein_attention.png")
'''
ax.text(0.05, 0.95, code_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.axis('off')
ax.set_title('Example Code (PyMOL)', fontsize=13, fontweight='bold')

# Alternative: Py3Dmol
ax = axes[1, 0]
code_text2 = '''# Example py3Dmol (Jupyter)
import py3Dmol

view = py3Dmol.view(width=800, height=600)
view.addModel(pdb_string, 'pdb')

# Color by attention×pLDDT
for i, val in enumerate(attn_plddt_values):
    color = rgb_from_value(val)
    view.setStyle({'resi': i+1}, 
                  {'cartoon': {'color': color}})

view.zoomTo()
view.show()
'''
ax.text(0.05, 0.95, code_text2, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.axis('off')
ax.set_title('Example Code (Py3Dmol)', fontsize=13, fontweight='bold')

# Interpretation guide
ax = axes[1, 1]
interp_text = '''BIOLOGICAL INTERPRETATION:

When creating 3D visualizations, look for:

✓ Clustering Patterns:
  • Do high attention×pLDDT residues cluster
    together in 3D space?
  
✓ Functional Regions:
  • Are they near active sites?
  • Do they overlap with binding interfaces?
  • Are they in structurally conserved regions?

✓ Secondary Structure:
  • Which structural elements (helices, sheets)
    receive high attention?

✓ Surface vs Core:
  • Are attention patterns surface-biased?
  • Or do they focus on structural core?

⚠ Important:
   Correlation with known functional sites
   strengthens mechanistic claims beyond
   pure statistical associations.
'''
ax.text(0.05, 0.95, interp_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
ax.axis('off')
ax.set_title('Interpretation Guide', fontsize=13, fontweight='bold')

fig4.suptitle('Optional Figure 4: 3D Structure Mapping\n(Requires PDB files and 3D visualization tools)',
              fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('fig4_3d_visualization_guide.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4 saved: fig4_3d_visualization_guide.png")
print("  ℹ This is a guide figure. Implement 3D visualization separately with PyMOL/Py3Dmol.")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY REPORT")
print("=" * 80)
print(f"\nFigures generated:")
print("  1. fig1_attention_plddt_heatmap.png - Heatmap across all proteins")
print("  2. fig2_correlation_per_head.png - Per-head correlation statistics")
print("  3. fig3_masking_performance.png - Performance vs. masking (SIMULATED)")
print("  4. fig4_3d_visualization_guide.png - 3D visualization setup guide")
print("\nNext steps for publication:")
print("  → Replace Fig 3 simulated data with actual model predictions")
print("  → Implement Fig 4 with real PDB structures and PyMOL/Py3Dmol")
print("  → Add baseline comparisons (standard GNN without attention)")
print("  → Perform statistical tests on masking experiment results")
print("  → Validate against known functional sites in proteins")
print("=" * 80)