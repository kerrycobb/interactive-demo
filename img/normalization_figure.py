import lamindb as ln
import numpy as np
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import issparse

af = ln.Artifact.connect("theislab/sc-best-practices").get(
    key="preprocessing_visualization/s4d8_quality_control.h5ad", is_latest=True
)
adata = af.load()
adata


scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0], legend=False)
axes[0].set_title("Raw")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1], legend=False)
axes[1].set_title("Normalized")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
fig.savefig("normalization_figure.png", dpi=150, bbox_inches="tight")

