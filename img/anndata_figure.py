import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# --- Layout constants ---
N_CELLS = 10       # rows in X / obs
N_GENES = 10       # cols in X / var
OBS_COLS = 3       # columns in obs (index, batch, cell_type)
VAR_ROWS = 2       # rows in var (index, gene)

CELL_W = 0.4       # width of each gene column in X (square cells)
CELL_H = CELL_W    # height = width for square cells

OBS_COL_W = CELL_W  # width of each obs column (square cells)
VAR_ROW_H = CELL_H # var rows same height as cells (square)

X_W = N_GENES * CELL_W
X_H = N_CELLS * CELL_H
OBS_W = OBS_COLS * OBS_COL_W
VAR_H = VAR_ROWS * VAR_ROW_H

GAP = 0.3          # gap between panels

# --- Origin positions ---
# X sits at (0, 0), growing right and up
x0, y0 = 0, 0

# obs is to the right of X
obs_x0 = x0 + X_W + GAP
obs_y0 = y0

# var is above X
var_x0 = x0
var_y0 = y0 + X_H + GAP

# --- Figure size ---
fig_w = X_W + GAP + OBS_W + 2.5
fig_h = VAR_H + GAP + X_H + 1.5

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_aspect("equal")
ax.axis("off")

# Helper: draw a filled rectangle with border and optional alpha
def draw_rect(ax, x, y, w, h, facecolor, edgecolor="black", lw=1.5, alpha=1.0, zorder=1):
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=lw, edgecolor=edgecolor,
        facecolor=facecolor, alpha=alpha, zorder=zorder
    )
    ax.add_patch(rect)

# Helper: draw faint internal gridlines
def draw_grid(ax, x0, y0, w, h, col_w, row_h, color="#cccccc", lw=0.4, zorder=2):
    n_cols = round(w / col_w)
    n_rows = round(h / row_h)
    for i in range(1, n_cols):
        ax.plot([x0 + i * col_w, x0 + i * col_w], [y0, y0 + h],
                color=color, lw=lw, zorder=zorder)
    for j in range(1, n_rows):
        ax.plot([x0, x0 + w], [y0 + j * row_h, y0 + j * row_h],
                color=color, lw=lw, zorder=zorder)

# ── X matrix ─────────────────────────────────────────────────────────────────
draw_rect(ax, x0, y0, X_W, X_H, facecolor="#dce8f5", edgecolor="#3a7abf", lw=2)
draw_grid(ax, x0, y0, X_W, X_H, CELL_W, CELL_H)
ax.text(x0 + X_W / 2, y0 + X_H / 2, "X",
        ha="center", va="center", fontsize=36, fontweight="bold",
        color="#1a4f8a", alpha=0.35, zorder=3)

# ── obs (cells × metadata columns) ───────────────────────────────────────────
draw_rect(ax, obs_x0, obs_y0, OBS_W, X_H, facecolor="#fde8d8", edgecolor="#d4631a", lw=2)
draw_grid(ax, obs_x0, obs_y0, OBS_W, X_H, OBS_COL_W, CELL_H)
ax.text(obs_x0 + OBS_W / 2, obs_y0 + X_H / 2, "obs",
        ha="center", va="center", fontsize=28, fontweight="bold",
        color="#9b3d0a", alpha=0.35, zorder=3)

# obs column header labels (rotated 90°)
obs_labels = ["index", "batch", "cell type"]
for i, label in enumerate(obs_labels):
    cx = obs_x0 + (i + 0.5) * OBS_COL_W
    cy = obs_y0 + X_H + 0.12
    ax.text(cx, cy, label, ha="center", va="bottom", fontsize=8.5,
            color="#9b3d0a", fontstyle="italic", zorder=4, rotation=90)

# ── var (gene metadata rows × genes) ─────────────────────────────────────────
draw_rect(ax, var_x0, var_y0, X_W, VAR_H, facecolor="#e2f5e2", edgecolor="#2e8b2e", lw=2)
draw_grid(ax, var_x0, var_y0, X_W, VAR_H, CELL_W, VAR_ROW_H)
ax.text(var_x0 + X_W / 2, var_y0 + VAR_H / 2, "var",
        ha="center", va="center", fontsize=28, fontweight="bold",
        color="#1a5c1a", alpha=0.35, zorder=3)

# var row labels (to the left of var panel) — gene first, then index
var_labels = ["gene", "index"]
for j, label in enumerate(var_labels):
    cy = var_y0 + (j + 0.5) * VAR_ROW_H
    cx = var_x0 - 0.15
    ax.text(cx, cy, label, ha="right", va="center", fontsize=8.5,
            color="#1a5c1a", fontstyle="italic", zorder=4)

# ── Dimension annotations (arrows + labels) ───────────────────────────────────
arrow_kw = dict(arrowstyle="<->", color="#555555", lw=1.2)
offset = 0.35

# cells arrow (left of X)
cells_x = x0 - offset
ax.annotate("", xy=(cells_x, y0 + X_H), xytext=(cells_x, y0),
            arrowprops=dict(**arrow_kw))
ax.text(cells_x - 0.15, y0 + X_H / 2, "cells",
        ha="right", va="center", fontsize=9, color="#555555", rotation=90)

# genes arrow (below X)
genes_y = y0 - offset
ax.annotate("", xy=(x0 + X_W, genes_y), xytext=(x0, genes_y),
            arrowprops=dict(**arrow_kw))
ax.text(x0 + X_W / 2, genes_y - 0.18, "genes",
        ha="center", va="top", fontsize=9, color="#555555")

# ── Set axis limits ───────────────────────────────────────────────────────────
margin = 1.2
ax.set_xlim(x0 - margin, obs_x0 + OBS_W + margin)
ax.set_ylim(y0 - margin, var_y0 + VAR_H + margin)

plt.tight_layout()
plt.savefig("anndata_diagram.svg", bbox_inches="tight")
plt.savefig("anndata_diagram.png", dpi=150, bbox_inches="tight")
