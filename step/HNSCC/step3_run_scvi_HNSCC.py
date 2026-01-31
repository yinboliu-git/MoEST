import os
import h5py
import numpy as np
import scvi
import anndata
from tqdm import tqdm
import gc

INPUT_H5 = "/path/to/input/DATASET_RAWCOUNTS.h5"
OUTPUT_H5 = "/path/to/output/DATASET_PROCESSED.h5"

LATENT_DIM = 30
EPOCHS = 50
BATCH_SIZE = 4096

expressions = []
batch_indices = []
section_names = []

with h5py.File(INPUT_H5, "r") as f:
    keys = [k for k in f.keys() if k.startswith("S")]
    keys.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    if "gene_names" in f:
        gene_names = f["gene_names"][:].astype(str)
    else:
        n_genes = f[keys[0]]["expression"].shape[1]
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    for i, k in enumerate(tqdm(keys)):
        if "expression" not in f[k]:
            continue
        expr = f[k]["expression"][:]
        expressions.append(expr)
        batch_indices.append(np.full(expr.shape[0], i))
        section_names.append(k)

X = np.concatenate(expressions, axis=0)
batch = np.concatenate(batch_indices, axis=0)

adata = anndata.AnnData(X=X)
adata.var_names = gene_names
adata.obs["batch"] = batch.astype(str)

del expressions, batch_indices
gc.collect()

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")

model = scvi.model.SCVI(
    adata,
    n_hidden=256,
    n_layers=3,
    n_latent=LATENT_DIM,
    gene_likelihood="nb",
)

model.train(
    max_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=True,
    early_stopping_patience=5,
)

os.makedirs("/path/to/checkpoints/scvi_teacher", exist_ok=True)
model.save("/path/to/checkpoints/scvi_teacher/model", overwrite=True)

latent_X = model.get_latent_representation(batch_size=BATCH_SIZE)

curr_idx = 0
with h5py.File(INPUT_H5, "r") as f_src, h5py.File(OUTPUT_H5, "w") as f_dst:
    if "gene_names" in f_src:
        f_src.copy("gene_names", f_dst)

    for k in tqdm(section_names):
        f_src.copy(k, f_dst)
        n = f_src[k]["expression"].shape[0]
        z = latent_X[curr_idx:curr_idx + n]
        curr_idx += n
        f_dst[k].create_dataset("scvi_latent", data=z.astype(np.float32))
