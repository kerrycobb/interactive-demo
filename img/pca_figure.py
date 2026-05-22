import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(42)

n_cells       = 50
n_genes       = 1000
n_pcs         = 20
n_cell_groups = 5

cells_per_group = n_cells // n_cell_groups   # 10
genes_per_group = n_genes // n_cell_groups   # 200 marker genes per group

cell_group = np.repeat(np.arange(n_cell_groups), cells_per_group)
gene_group = np.repeat(np.arange(n_cell_groups), genes_per_group)

# ── Realistic single-cell count matrix ───────────────────────────────────────
# Negative-binomial counts: sparse, overdispersed, heavily zero-inflated
# mu and r control mean expression and dispersion (p = r / (r + mu))
def nb_counts(mu, r, size, rng):
    p = r / (r + mu)
    return rng.negative_binomial(r, p, size=size)

# Start with near-zero background for all genes in all cells
gene_matrix = nb_counts(mu=0.05, r=0.5, size=(n_cells, n_genes), rng=rng).astype(float)

# Housekeeping genes (~5 %): moderate expression everywhere
hk_idx = rng.choice(n_genes, size=50, replace=False)
gene_matrix[:, hk_idx] = nb_counts(mu=8, r=2, size=(n_cells, 50), rng=rng)

# Cell-type marker genes: elevated in primary group, partially expressed in neighbours
# Cell-type marker genes: soft group membership with random per-cell variability
# Each cell gets a random blend weight toward neighbouring groups
for cg in range(n_cell_groups):
    cell_idx  = np.where(cell_group == cg)[0]
    marker_idx = np.where(gene_group == cg)[0]

    for ci in cell_idx:
        # Random cell-level scaling: some cells in a group express markers weakly
        cell_scale = rng.uniform(0.4, 1.4)
        mu_primary = 8 * cell_scale
        gene_matrix[ci, marker_idx] = nb_counts(
            mu=mu_primary, r=1.2, size=(len(marker_idx),), rng=rng
        )

    # All other groups also express these markers at decaying levels with noise
    for other_cg in range(n_cell_groups):
        if other_cg == cg:
            continue
        ring_dist = min(
            abs(other_cg - cg),
            n_cell_groups - abs(other_cg - cg)
        )
        # Exponential decay with distance + per-group random jitter
        mu_leak = rng.uniform(0.5, 2.0) * np.exp(-ring_dist * 0.8)
        other_idx = np.where(cell_group == other_cg)[0]
        gene_matrix[np.ix_(other_idx, marker_idx)] += nb_counts(
            mu=mu_leak, r=0.8, size=(len(other_idx), len(marker_idx)), rng=rng
        )

# Log1p-normalise for display (as is standard before PCA)
gene_matrix_display = np.log1p(gene_matrix)

# ── PC matrix: derived from actual PCA on the normalised counts ───────────────
# Mean-centre before PCA (standard scanpy workflow)
X_scaled = StandardScaler(with_std=False).fit_transform(gene_matrix_display)
pca = PCA(n_components=n_pcs, random_state=42)
pc_matrix = pca.fit_transform(X_scaled)

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 5))
gs = gridspec.GridSpec(
    1, 2,
    width_ratios=[6, 1],
    wspace=0.12,
)

ax_genes = fig.add_subplot(gs[0])
ax_pcs   = fig.add_subplot(gs[1])

for ax, data, xlabel, cmap in [
    (ax_genes, gene_matrix_display, "Genes",  "YlOrRd"),
    (ax_pcs,   pc_matrix,           "PCs",    "RdBu_r"),
]:
    ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(xlabel, fontsize=13, labelpad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor("N")

# Draw images separately so we can keep im references for colorbars
im_genes = ax_genes.images[0]
im_pcs   = ax_pcs.images[0]

# width_ratios=[6,1] → PC panel is 1/6 the width of genes panel.
# To match physical widths: genes cbar = 16.67 % of genes panel width, centered.
# axes coords: [x0, y0, width, height]  (y0 negative = below panel)
cax_genes = ax_genes.inset_axes([0.4167, -0.10, 0.1667, 0.04],
                                 transform=ax_genes.transAxes)
cbar_genes = plt.colorbar(im_genes, cax=cax_genes, orientation="horizontal")
cbar_genes.set_label("Counts", fontsize=11)
cbar_genes.set_ticks([])

cax_pcs = ax_pcs.inset_axes([0.0, -0.10, 1.0, 0.04],
                              transform=ax_pcs.transAxes)
cbar_pcs = plt.colorbar(im_pcs, cax=cax_pcs, orientation="horizontal")
cbar_pcs.set_label("PC Score", fontsize=11)
cbar_pcs.set_ticks([])

# Shared "Cells" y-axis label on the left panel
ax_genes.set_ylabel("Cells", fontsize=13, labelpad=8)
ax_genes.yaxis.set_label_position("left")

plt.savefig(os.path.join(HERE, "pca_figure.png"), dpi=150, bbox_inches="tight")
plt.show()
