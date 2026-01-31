import os
import h5py
import numpy as np
import scvi
import anndata
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

INPUT_H5 = "/path/to/input/MISAR_XYT.h5"
OUTPUT_H5 = "/path/to/output/MISAR_XYT_PROCESSED.h5"

LATENT_DIM = 30
EPOCHS = 100

if not os.path.exists(INPUT_H5):
    raise FileNotFoundError("Input file not found")

expressions = []
batch_indices = []
sample_names = []

with h5py.File(INPUT_H5, "r") as f:
    if "gene_names" in f:
        gene_names = f["gene_names"][:].astype(str)
    else:
        first_key = [k for k in f.keys() if k.startswith("E")][0]
        n_genes = f[first_key]["expression"].shape[1]
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    keys = [k for k in f.keys() if k.startswith("E")]
    keys.sort()

    for i, k in enumerate(tqdm(keys)):
        expr = f[k]["expression"][:]
        expressions.append(expr)
        batch_indices.append(np.full(expr.shape[0], i))
        sample_names.append(k)

X = np.concatenate(expressions, axis=0)
batch = np.concatenate(batch_indices, axis=0)

adata = anndata.AnnData(X=X)
adata.var_names = gene_names
adata.obs["batch"] = batch.astype(str)

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata, n_hidden=128, n_layers=2, n_latent=LATENT_DIM)
model.train(max_epochs=EPOCHS, early_stopping=True)

os.makedirs("/path/to/checkpoints/scvi_teacher", exist_ok=True)
model.save("/path/to/checkpoints/scvi_teacher/model", overwrite=True)

latent_X = model.get_latent_representation()

curr = 0
with h5py.File(INPUT_H5, "r") as f_src, h5py.File(OUTPUT_H5, "w") as f_dst:
    for k in f_src.keys():
        if not k.startswith("E"):
            f_src.copy(k, f_dst)

    for k in tqdm(keys):
        f_src.copy(k, f_dst)
        grp_dst = f_dst[k]
        n = grp_dst["expression"].shape[0]
        z = latent_X[curr:curr + n]
        curr += n
        if "scvi_latent" in grp_dst:
            del grp_dst["scvi_latent"]
        grp_dst.create_dataset("scvi_latent", data=z.astype(np.float32))
