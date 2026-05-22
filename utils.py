import numpy as np
from sklearn.neighbors import NearestNeighbors


def median_rank_shift(
    X1,
    X2,
    k=30,
    max_search=None,
):
    """
    Compute the median rank shift metric from:
    Antonsson et al. 2025 https://doi.org/10.1101/gr.279886.124

    Parameters
    ----------
    X1: Original embedding.
    X2: Corrected embedding (same cells, same row order).
    k: Number of nearest neighbours to evaluate.
    max_search: Number of neighbors from X2 

    Returns
    -------
    scores: Per-cell median rank shift.
    """
    n_cells = X1.shape[0]

    if X2.shape[0] != n_cells:
        raise ValueError("X1 and X2 must contain the same number of cells")
    if k >= n_cells:
        raise ValueError("k must be smaller than the number of cells")

    if max_search is None:
        max_search = min(n_cells - 1, max(500, 10 * k))
    else:
        max_search = min(n_cells - 1, int(max_search))

    idx1 = (
        NearestNeighbors(n_neighbors=k, metric="euclidean")
        .fit(X1)
        .kneighbors(return_distance=False)
    )  # (n_cells, k)

    idx2 = (
        NearestNeighbors(n_neighbors=max_search, metric="euclidean")
        .fit(X2)
        .kneighbors(return_distance=False)
    )  # (n_cells, max_search)

    penalty = max_search + 1
    original_ranks = np.arange(1, k + 1)
    scores = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        # Map neighbour index to rank (1-based) in X2 for cell i
        rank_map = {cell: rank for rank, cell in enumerate(idx2[i], start=1)}
        corrected_ranks = np.array([rank_map.get(j, penalty) for j in idx1[i]], dtype=float)
        scores[i] = np.median(np.abs(original_ranks - corrected_ranks))
    return scores



