import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Cell colors: index 0 = cell 1 (rank 1 = closest), index 9 = cell 10 (furthest)
N = 10
original_order = list(range(1, N + 1))
# 3 pairs swapped: 1↔3, 2↔8, 4↔9  → 6/10 cells shift → median rank shift = 2
corrected_order = [3, 8, 1, 9, 5, 6, 7, 2, 4, 10]

colors = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#6a994e",
]

# ── Scatter layout ────────────────────────────────────────────────────────────
base_dist = 1.2
distances = [base_dist * rank for rank in range(1, N + 1)]

def make_pos(order, seed):
    """Place cells at rank-proportional distances with independent random angles."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    return {
        cell: (distances[rank - 1] * np.cos(angles[rank - 1]),
               distances[rank - 1] * np.sin(angles[rank - 1]))
        for rank, cell in enumerate(order, start=1)
    }

orig_pos_xy = make_pos(original_order, seed=42)
corr_pos_xy = make_pos(corrected_order, seed=99)


def draw_scatter(ax, pos_xy, title):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, color="#333", pad=8)

    # Concentric distance rings
    for rank in [1, 5, 10]:
        circle = plt.Circle((0, 0), base_dist * rank,
                             fill=False, linestyle="--", linewidth=0.8, color="#ccc", zorder=0)
        ax.add_patch(circle)

    # Neighbor dots
    for cell, (x, y) in pos_xy.items():
        ax.scatter(x, y, s=350, color=colors[cell - 1], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(x, y, str(cell), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    # Reference cell (star only, no label)
    ax.scatter(0, 0, s=500, marker="*", color="black", zorder=5,
               edgecolors="white", linewidths=1)

    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)


# ── Figure 1: scatter plots (pre and post correction) ─────────────────────────
fig1, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(14, 7))
draw_scatter(ax_pre,  orig_pos_xy, "Before correction")
draw_scatter(ax_post, corr_pos_xy, "After correction")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "nn_scatter_fig.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2: rank-order rows with arrows ─────────────────────────────────────
cell_size = 1.0
gap = 0.15
row_gap = 2.2

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.set_aspect("equal")
ax2.axis("off")


def draw_row(ax, order, y, label):
    for i, cell_num in enumerate(order):
        x = i * (cell_size + gap)
        color = colors[cell_num - 1]
        rect = mpatches.FancyBboxPatch(
            (x, y), cell_size, cell_size,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + cell_size / 2, y + cell_size / 2, str(cell_num),
                ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(-0.6, y + cell_size / 2, label,
            ha="right", va="center", fontsize=11, color="#333")


y_orig = row_gap
y_corr = 0

draw_row(ax2, original_order, y_orig, "Original\nranks")
draw_row(ax2, corrected_order, y_corr, "Corrected\nranks")

shifts = [
    (0, corrected_order.index(1),  0.35),
    (1, corrected_order.index(2),  0.5),
    (2, corrected_order.index(3), -0.35),
    (3, corrected_order.index(4),  0.25),
    (7, corrected_order.index(8), -0.5),
    (8, corrected_order.index(9), -0.25),
]

for orig_p, corr_p, _ in shifts:
    x_orig = orig_p * (cell_size + gap) + cell_size / 2
    x_corr = corr_p * (cell_size + gap) + cell_size / 2
    arrow = FancyArrowPatch(
        (x_orig, y_orig),
        (x_corr, y_corr + cell_size),
        arrowstyle="-|>", color="#333", linewidth=1.5,
        mutation_scale=14, linestyle="dashed",
        connectionstyle="arc3,rad=0.25",
    )
    ax2.add_patch(arrow)

total_width = N * (cell_size + gap) - gap
ax2.set_xlim(-1.2, total_width + 0.3)
ax2.set_ylim(-0.5, y_orig + cell_size + 0.5)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "median_rank_shift_fig.png"), dpi=150, bbox_inches="tight")
plt.show()
