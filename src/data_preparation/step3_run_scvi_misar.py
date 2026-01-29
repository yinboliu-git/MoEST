import os
import h5py
import numpy as np
import scanpy as sc
import scvi
import torch
import anndata
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= Configuration =================
# Use environment variable or default relative path
# Input: File containing Image Features just created
INPUT_H5 = os.environ.get('INPUT_H5', './data/Misar/MISAR_4D_Final_Fixed.h5')
# Output: Final training file containing Latent Codes
OUTPUT_H5 = os.environ.get('OUTPUT_H5', './data/Misar/MISAR_4D_Processed_Fixed.h5')

LATENT_DIM = 30
EPOCHS = 100

print(f"[Step 3] Training scVI Teacher...")

# 1. Prepare data
print("   Reading Expression Data...")
expressions = []
batch_indices = []
sample_names = []

with h5py.File(INPUT_H5, 'r') as f:
    if 'gene_names' in f:
        gene_names = f['gene_names'][:].astype(str)
    else:
        # Fallback
        first_key = [k for k in f.keys() if k.startswith('E')][0]
        n_genes = f[first_key]['expression'].shape[1]
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    keys = [k for k in f.keys() if k.startswith('E')]
    keys.sort()

    for i, k in enumerate(tqdm(keys)):
        expr = f[k]['expression'][:]
        expressions.append(expr)
        # Use sample index as Batch ID (remove inter-sample batch effects)
        batch_indices.append(np.full(expr.shape[0], i))
        sample_names.append(k)

X = np.concatenate(expressions, axis=0)
batch = np.concatenate(batch_indices, axis=0)

# 2. scVI training
print(f"   Training scVI on {X.shape[0]} cells...")
adata = anndata.AnnData(X=X)
adata.var_names = gene_names
adata.obs['batch'] = batch.astype(str)

scvi.model.SCVI.setup_anndata(adata, batch_key='batch')
model = scvi.model.SCVI(adata, n_hidden=128, n_layers=2, n_latent=LATENT_DIM)
model.train(max_epochs=EPOCHS, early_stopping=True)

# Save model (for inference)
os.makedirs("checkpoints_misar_scvi", exist_ok=True)
model.save("checkpoints_misar_scvi/teacher_model", overwrite=True)

# Get Latent
latent_X = model.get_latent_representation()

# 3. Write to new file
print(f"   Saving to {OUTPUT_H5}...")
curr = 0
with h5py.File(INPUT_H5, 'r') as f_src, h5py.File(OUTPUT_H5, 'w') as f_dst:
    # Copy global information
    if 'gene_names' in f_src: f_src.copy('gene_names', f_dst)

    for i, k in enumerate(tqdm(keys)):
        # Copy original group
        grp_src = f_src[k]
        grp_dst = f_dst.create_group(k)

        for dset in grp_src.keys():
            grp_src.copy(dset, grp_dst)

        # Write Latent
        n = grp_src['expression'].shape[0]
        z = latent_X[curr : curr+n]
        curr += n

        grp_dst.create_dataset("scvi_latent", data=z.astype(np.float32))

print("Teacher Step Done! Ready for Student Training.")
