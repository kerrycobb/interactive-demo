import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng()
n_points = 20
k = 3

points = rng.uniform(0, 10, size=(n_points, 2))

nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)  # +1 because point is its own neighbour
distances, indices = nbrs.kneighbors(points)

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.axis("off")

# Draw edges
drawn = set()
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:  # skip self (index 0)
        edge = tuple(sorted((i, j)))
        if edge not in drawn:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                color="#adb5bd", linewidth=1, zorder=1,
            )
            drawn.add(edge)

# Draw points
ax.scatter(points[:, 0], points[:, 1], s=200, color="#4361ee",
           edgecolors="white", linewidths=1.5, zorder=2)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "knn_figure.png"), dpi=150, bbox_inches="tight")
# plt.show()
