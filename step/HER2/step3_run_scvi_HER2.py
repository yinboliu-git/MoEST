```python
import os
import h5py
import numpy as np
import scvi
import anndata
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

INPUT_H5 = "/path/to/input/DATASET.h5"
OUTPUT_H5 = "/path/to/output/DATASET_PROCESSED.h5"

LATENT_DIM = 30
EPOCHS = 100
CKPT_DIR = "/path/to/checkpoints/scvi_teacher"

expressions = []
batch_ids = []
sections_order = []

with h5py.File(INPUT_H5, 'r') as f:
    sections = [k for k in f.keys() if k and k[0].isalpha() and 'gene' not in k]
    sections.sort()

    if 'gene_names' in f:
        gene_names = f['gene_names'][:].astype(str)
    else:
        gene_names = np.arange(f[sections[0]]['expression'].shape[1]).astype(str)

    for i, sec in enumerate(tqdm(sections, desc="Reading")):
        if 'expression' not in f[sec]:
            continue
        expr = f[sec]['expression'][:]
        expressions.append(expr)
        n = expr.shape[0]
        batch_ids.append(np.full(n, i))
        sections_order.extend([sec] * n)

X = np.concatenate(expressions, axis=0)
batch = np.concatenate(batch_ids, axis=0).astype(str)

adata = anndata.AnnData(X=X)
adata.var_names = gene_names
adata.obs["batch"] = batch

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata, n_hidden=128, n_layers=2, n_latent=LATENT_DIM)
model.train(max_epochs=EPOCHS, early_stopping=True)

os.makedirs(CKPT_DIR, exist_ok=True)
model.save(os.path.join(CKPT_DIR, "model"), overwrite=True)

latent = model.get_latent_representation()

idx = 0
with h5py.File(INPUT_H5, 'r') as f_src, h5py.File(OUTPUT_H5, 'w') as f_dst:
    if 'gene_names' in f_src:
        f_src.copy('gene_names', f_dst)

    for sec in tqdm(sections, desc="Writing"):
        if 'expression' not in f_src[sec]:
            continue

        n = f_src[sec]['expression'].shape[0]
        z = latent[idx:idx + n]
        idx += n

        grp = f_dst.create_group(sec)
        for k in f_src[sec].keys():
            f_src[sec].copy(k, grp)

        grp.create_dataset("scvi_latent", data=z.astype(np.float32))
```
