import os
import re
import random
import argparse
import warnings
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scanpy as sc
import squidpy as sq
import anndata as ad
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "/path/to/data/DATASET_PROCESSED.h5"
BASE_SAVE_DIR = "/path/to/output/checkpoints"
SVG_CACHE_PATH = "/path/to/cache/svg_cache.npz"

NUM_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
PATIENCE = 15
NUM_SVGS = 2000

DOWNSAMPLE_RATE = 1
MIN_COUNTS_THRESHOLD = 50

LAMBDA_NB = 1.0
LAMBDA_ALIGN = 10.0
LAMBDA_GRAD = 0.1

DIM_UNI = 1024
DIM_HIDDEN = 256
SCVI_LATENT_DIM = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_global_z_map(all_sections):
    def extract_num(s):
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 0

    sorted_sections = sorted(all_sections, key=extract_num)
    n = len(sorted_sections)
    z_map = {}
    if n <= 1:
        z_map[sorted_sections[0]] = 0.0
    else:
        for i, sid in enumerate(sorted_sections):
            z_map[sid] = float(i) / float(n - 1)
    return z_map

def select_spatially_variable_genes_cached(h5_path, train_sections, n_top_genes=2000, cache_path=SVG_CACHE_PATH):
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["indices"], data["names"]
        except Exception:
            pass

    gene_counter = {}
    with h5py.File(h5_path, "r") as f:
        first_key = train_sections[0]
        if "gene_names" in f[first_key]:
            all_genes_raw = f[first_key]["gene_names"][:].astype(str)
        else:
            all_genes_raw = f["gene_names"][:].astype(str)

    rng = random.Random(GLOBAL_SEED)
    subset_samples = train_sections if len(train_sections) < 10 else rng.sample(train_sections, 10)

    for sid in tqdm(subset_samples):
        try:
            with h5py.File(h5_path, "r") as f:
                if "expression" not in f[sid]:
                    continue
                counts = f[sid]["expression"][:]
                coords = f[sid]["coords_3d"][:, :2]

            adata = ad.AnnData(X=counts)
            adata.var_names = all_genes_raw
            adata.obsm["spatial"] = coords

            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_THRESHOLD)

            if adata.n_obs > 5000:
                sc.pp.subsample(adata, n_obs=5000)

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            if adata.n_vars < 50:
                continue

            sq.gr.spatial_neighbors(adata, coord_type="generic")
            sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=10)

            res = adata.uns.get("moranI", None)
            sort_col = "I" if "I" in res.columns else "moranI"
            top_genes = res.sort_values(sort_col, ascending=False).head(n_top_genes).index.tolist()
            for g in top_genes:
                gene_counter[g] = gene_counter.get(g, 0) + 1
        except Exception:
            continue

    if not gene_counter:
        final_indices = np.arange(n_top_genes)
        final_svgs = all_genes_raw[:n_top_genes]
    else:
        sorted_genes = sorted(gene_counter.items(), key=lambda x: x[1], reverse=True)
        final_svgs = [g[0] for g in sorted_genes[:n_top_genes]]
        gene_map = {name: i for i, name in enumerate(list(all_genes_raw))}
        final_indices = np.array([gene_map[g] for g in final_svgs if g in gene_map])

    np.savez(cache_path, indices=final_indices, names=final_svgs)
    return final_indices, final_svgs

class OpenSTDataset_Ram(Dataset):
    def __init__(self, h5_path, gene_indices, allowed_sections, z_map, downsample_rate=DOWNSAMPLE_RATE):
        self.h5_path = h5_path
        self.gene_indices = gene_indices

        meta_list = []
        with h5py.File(h5_path, "r") as f:
            for sid in allowed_sections:
                if sid not in f:
                    continue
                grp = f[sid]
                if "valid_mask" in grp:
                    indices = np.where(grp["valid_mask"][:].astype(bool))[0]
                else:
                    expr = grp["expression"][:]
                    counts = expr.sum(1)
                    indices = np.where(counts > MIN_COUNTS_THRESHOLD)[0]
                for idx in indices:
                    meta_list.append((sid, idx))

        if downsample_rate < 1.0:
            rng = random.Random(GLOBAL_SEED)
            rng.shuffle(meta_list)
            meta_list = meta_list[: int(len(meta_list) * downsample_rate)]

        self.data_vis = []
        self.data_pos = []
        self.data_grad = []
        self.data_y = []
        self.data_lib_size = []
        self.data_rna_z = []

        meta_dict = {}
        for sid, idx in meta_list:
            meta_dict.setdefault(sid, []).append(idx)

        with h5py.File(h5_path, "r") as f:
            for sid, idxs in meta_dict.items():
                grp = f[sid]
                idxs = sorted(idxs)

                self.data_vis.append(grp["uni_features"][idxs])

                coords = grp["coords_3d"][idxs]
                xy = coords[:, :2].astype(np.float32)
                xy_min, xy_max = xy.min(0), xy.max(0)
                xy_norm = (xy - xy_min) / (xy_max - xy_min + 1e-6)

                z_col = np.full((len(idxs), 1), z_map.get(sid, 0.0), dtype=np.float32)
                self.data_pos.append(np.concatenate([xy_norm, z_col], axis=1))

                if "sobel_gradients" in grp:
                    g = grp["sobel_gradients"][:]
                    if g.ndim > 1:
                        g = np.mean(g, axis=(1, 2))
                    self.data_grad.append(g[idxs])
                else:
                    self.data_grad.append(np.zeros(len(idxs)))

                expr = grp["expression"][idxs]
                self.data_lib_size.append(expr.sum(1))
                self.data_y.append(expr[:, self.gene_indices])

                if "scvi_latent" in grp:
                    self.data_rna_z.append(grp["scvi_latent"][idxs])
                else:
                    self.data_rna_z.append(np.zeros((len(idxs), SCVI_LATENT_DIM)))

        self.data_vis = torch.tensor(np.concatenate(self.data_vis), dtype=torch.float32)
        self.data_pos = torch.tensor(np.concatenate(self.data_pos), dtype=torch.float32)
        self.data_grad = torch.tensor(np.concatenate(self.data_grad), dtype=torch.float32).unsqueeze(1)
        self.data_y = torch.tensor(np.concatenate(self.data_y), dtype=torch.float32)
        self.data_lib_size = torch.tensor(np.concatenate(self.data_lib_size), dtype=torch.float32).unsqueeze(1)
        self.data_rna_z = torch.tensor(np.concatenate(self.data_rna_z), dtype=torch.float32)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return {
            "vis": self.data_vis[idx],
            "pos": self.data_pos[idx],
            "grad": self.data_grad[idx],
            "y": self.data_y[idx],
            "library_size": self.data_lib_size[idx],
            "rna_z": self.data_rna_z[idx],
        }

class LearnableFourierEncoding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=128, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SparseMoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=1, dropout_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim + 1, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim * 4, output_dim),
                nn.Dropout(dropout_rate),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x, grad):
        router_input = torch.cat([x, grad], dim=-1)
        logits = self.router(router_input)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        output = torch.zeros(x.size(0), x.size(1), device=x.device)
        for i in range(self.top_k):
            idx = topk_indices[:, i]
            val = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    output[mask] += val[mask] * self.experts[e](x[mask])
        return output, probs

class MoEST_OpenST(nn.Module):
    def __init__(self, num_genes, dim_vis=1024, dim_hidden=256, num_experts=4, dropout_rate=0.3):
        super().__init__()
        self.img_enc = nn.Linear(dim_vis, dim_hidden)
        self.pos_enc = nn.Sequential(
            LearnableFourierEncoding(3, 128, 10.0),
            nn.Linear(256, dim_hidden),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.spatial_ctx = nn.Identity()
        self.moe = SparseMoELayer(dim_hidden, dim_hidden, num_experts, 1, 0.2)

        self.gene_decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, num_genes * 2),
        )

        self.align_projector = nn.Sequential(
            nn.Linear(dim_hidden, 128),
            nn.GELU(),
            nn.Linear(128, SCVI_LATENT_DIM),
        )

    def forward(self, vis, pos, grad, library_size, apply_msm=False):
        if apply_msm and self.training:
            vis = vis * (torch.rand(vis.size(0), 1, device=vis.device) > 0.2)

        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.dropout(z)
        z_moe, _ = self.moe(z, grad)
        z = z + z_moe

        preds = self.gene_decoder(z).view(z.size(0), -1, 2)
        mu = F.softplus(preds[:, :, 0]) * library_size + 1e-6
        theta = F.softplus(preds[:, :, 1]) + 1e-6
        return mu, theta, z, self.align_projector(z)

class NBLoss(nn.Module):
    def forward(self, y, mu, theta):
        eps = 1e-8
        return torch.mean(
            torch.lgamma(y + theta + eps)
            - torch.lgamma(theta + eps)
            - torch.lgamma(y + 1.0)
            + theta * torch.log(theta / (theta + mu + eps) + eps)
            + y * torch.log(mu / (theta + mu + eps) + eps)
        )

def coupled_gradient_loss(g, pos, sobel_grad):
    idx = torch.randperm(g.size(0))
    d_g = torch.abs(g - g[idx])
    boundary = torch.maximum(sobel_grad, sobel_grad[idx])
    weight = torch.exp(-5.0 * boundary)
    d_pos = torch.norm(pos - pos[idx], dim=1, keepdim=True)
    mask = (d_pos < 1.0).float()
    return (weight * d_g * mask).mean()

def compute_metrics(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            mu, _, _, _ = model(
                batch["vis"].to(device),
                batch["pos"].to(device),
                batch["grad"].to(device),
                batch["library_size"].to(device),
            )
            preds.append(mu.cpu().numpy())
            targets.append(batch["y"].cpu().numpy())

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    p_log = np.log1p(p)
    t_log = np.log1p(t)

    mse = np.mean((p_log - t_log) ** 2)
    pcc, scc = [], []
    for i in range(p.shape[1]):
        if np.var(t_log[:, i]) > 1e-9:
            pv, _ = pearsonr(p_log[:, i], t_log[:, i])
            sv, _ = spearmanr(p_log[:, i], t_log[:, i])
            if not np.isnan(pv):
                pcc.append(pv)
            if not np.isnan(sv):
                scc.append(sv)

    return {
        "MSE": mse,
        "PCC": float(np.mean(pcc)),
        "SCC": float(np.mean(scc)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(GLOBAL_SEED)
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    with h5py.File(DATA_PATH, "r") as f:
        all_sections = [k for k in f.keys() if k.startswith("S")]

    all_sections.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    global_z_map = build_global_z_map(all_sections)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
    results = {"MSE": [], "PCC": [], "SCC": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_sections)):
        train_pool = [all_sections[i] for i in train_idx]
        test_pool = [all_sections[i] for i in test_idx]
        train_inner, val_inner = train_test_split(train_pool, test_size=0.1, random_state=GLOBAL_SEED + fold)

        svg_indices, _ = select_spatially_variable_genes_cached(DATA_PATH, train_inner, NUM_SVGS)

        train_ds = OpenSTDataset_Ram(DATA_PATH, svg_indices, train_inner, global_z_map)
        val_ds = OpenSTDataset_Ram(DATA_PATH, svg_indices, val_inner, global_z_map)
        test_ds = OpenSTDataset_Ram(DATA_PATH, svg_indices, test_pool, global_z_map)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=8)

        model = MoEST_OpenST(NUM_SVGS, DIM_UNI, DIM_HIDDEN)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        nb_loss = NBLoss()
        align_loss = nn.MSELoss()

        best_pcc = -1.0
        best_path = os.path.join(BASE_SAVE_DIR, f"fold{fold}.pth")
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                mu, theta, _, z_s = model(
                    batch["vis"].to(DEVICE),
                    batch["pos"].to(DEVICE),
                    batch["grad"].to(DEVICE),
                    batch["library_size"].to(DEVICE),
                    apply_msm=True,
                )
                loss = (
                    LAMBDA_NB * nb_loss(batch["y"].to(DEVICE), mu, theta)
                    + LAMBDA_ALIGN * align_loss(z_s, batch["rna_z"].to(DEVICE))
                    + LAMBDA_GRAD * coupled_gradient_loss(mu, batch["pos"].to(DEVICE), batch["grad"].to(DEVICE))
                )
                loss.backward()
                optimizer.step()

            scheduler.step()
            val_res = compute_metrics(model, val_loader, DEVICE)
            if val_res["PCC"] > best_pcc:
                best_pcc = val_res["PCC"]
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        model.load_state_dict(torch.load(best_path))
        test_res = compute_metrics(model, test_loader, DEVICE)
        for k in results:
            results[k].append(test_res[k])

    print(
        np.mean(results["MSE"]),
        np.mean(results["PCC"]),
        np.mean(results["SCC"]),
    )
