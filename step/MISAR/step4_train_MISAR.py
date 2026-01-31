import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import random
import argparse
import warnings
import scanpy as sc
import squidpy as sq
import anndata as ad
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

DATA_PATH = "/path/to/data/MISAR_XYT_Processed.h5"
BASE_SAVE_DIR = "/path/to/checkpoints/misar"

NUM_FOLDS = 5
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
PATIENCE = 15
NUM_SVGS = 2000

MIN_COUNTS_THRESHOLD = 20

LAMBDA_NB = 10.0
LAMBDA_ALIGN = 5.0
LAMBDA_GRAD = 0.1

DIM_UNI = 1024
DIM_HIDDEN = 256
SCVI_LATENT_DIM = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_spatially_variable_genes(h5_path, n_top_genes=2000, cache_path="svg_cache_misar.npz"):
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if len(data["indices"]) > 0:
            return data["indices"], data["names"]

    gene_counter = {}
    with h5py.File(h5_path, "r") as f:
        samples = [k for k in f.keys() if k.startswith("E")]
        if "gene_names" in f:
            gene_names = f["gene_names"][:].astype(str)
        else:
            gene_names = f[samples[0]]["gene_names"][:].astype(str)

    subset = samples if len(samples) < 5 else random.sample(samples, 5)

    for sid in subset:
        try:
            with h5py.File(h5_path, "r") as f:
                counts = f[sid]["expression"][:]
                coords = f[sid]["coords_xyt"][:, :2]

            adata = ad.AnnData(X=counts)
            adata.var_names = gene_names
            adata.obsm["spatial"] = coords

            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_THRESHOLD)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            if adata.n_vars < 50:
                continue

            sq.gr.spatial_neighbors(adata, coord_type="generic")
            sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=10)

            res = adata.uns["moranI"]
            col = "I" if "I" in res.columns else "moranI"
            top = res.sort_values(col, ascending=False).head(n_top_genes).index.tolist()

            for g in top:
                gene_counter[g] = gene_counter.get(g, 0) + 1
        except Exception:
            continue

    if not gene_counter:
        return np.arange(n_top_genes), gene_names[:n_top_genes]

    sorted_genes = sorted(gene_counter.items(), key=lambda x: x[1], reverse=True)
    selected = [g[0] for g in sorted_genes[:n_top_genes]]
    gene_map = {g: i for i, g in enumerate(gene_names)}
    indices = np.array([gene_map[g] for g in selected if g in gene_map])

    np.savez(cache_path, indices=indices, names=selected)
    return indices, selected


class MisarSectionDataset(Dataset):
    def __init__(self, h5_path, gene_indices, allowed_sections=None):
        self.data_vis = []
        self.data_pos = []
        self.data_grad = []
        self.data_y = []
        self.data_lib_size = []
        self.data_rna_z = []

        with h5py.File(h5_path, "r") as f:
            keys = [k for k in f.keys() if k.startswith("E")]
            if allowed_sections:
                keys = [k for k in keys if k in allowed_sections]
            keys.sort()

            for k in keys:
                if "uni_features" not in f[k]:
                    continue

                expr = f[k]["expression"][:]
                counts = expr.sum(1)
                mask = counts > MIN_COUNTS_THRESHOLD
                if not np.any(mask):
                    continue

                self.data_vis.append(f[k]["uni_features"][mask])

                coords = f[k]["coords_xyt"][mask]
                xy = coords[:, :2].astype(np.float32)
                t = coords[:, 2:].astype(np.float32)

                xy_min, xy_max = xy.min(0), xy.max(0)
                xy_norm = (xy - xy_min) / (xy_max - xy_min + 1e-6)
                t_norm = t / 20.0
                self.data_pos.append(np.concatenate([xy_norm, t_norm], axis=1))

                if "sobel_gradients" in f[k]:
                    g = f[k]["sobel_gradients"][mask]
                    g = np.mean(g, axis=(1, 2)) if g.ndim > 1 else g
                else:
                    g = np.zeros(np.sum(mask))
                self.data_grad.append(g)

                self.data_y.append(expr[mask][:, gene_indices])
                self.data_lib_size.append(counts[mask])

                if "scvi_latent" in f[k]:
                    self.data_rna_z.append(f[k]["scvi_latent"][mask])
                else:
                    self.data_rna_z.append(np.zeros((np.sum(mask), SCVI_LATENT_DIM)))

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
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(input_dim * 4, output_dim),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x, grad):
        logits = self.router(torch.cat([x, grad], dim=-1))
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        out = torch.zeros(x.size(0), x.size(1), device=x.device)
        for i in range(self.top_k):
            idx = topk_indices[:, i]
            val = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += val[mask] * self.experts[e](x[mask])
        return out, probs


class MoEST_Misar(nn.Module):
    def __init__(self, num_genes):
        super().__init__()
        self.img_enc = nn.Linear(DIM_UNI, DIM_HIDDEN)
        self.pos_enc = nn.Sequential(
            LearnableFourierEncoding(3, 128),
            nn.Linear(256, DIM_HIDDEN),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.moe = SparseMoELayer(DIM_HIDDEN, DIM_HIDDEN)
        self.decoder = nn.Sequential(
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.LayerNorm(DIM_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DIM_HIDDEN, num_genes * 2),
        )
        self.align = nn.Sequential(
            nn.Linear(DIM_HIDDEN, 128),
            nn.GELU(),
            nn.Linear(128, SCVI_LATENT_DIM),
        )
        self.func = nn.Sequential(
            nn.Linear(DIM_HIDDEN, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, vis, pos, grad, lib, apply_msm=False):
        if apply_msm and self.training:
            vis = vis * (torch.rand(vis.size(0), 1, device=vis.device) > 0.2)

        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.dropout(z)
        z_moe, _ = self.moe(z, grad)
        z = z + z_moe

        pred = self.decoder(z).view(z.size(0), -1, 2)
        mu = F.softplus(pred[:, :, 0]) * lib + 1e-6
        theta = F.softplus(pred[:, :, 1]) + 1e-6

        return mu, theta, self.func(z), self.align(z)


class NBLoss(nn.Module):
    def forward(self, y, mu, theta):
        eps = 1e-8
        return torch.mean(
            torch.lgamma(theta + eps)
            + torch.lgamma(y + 1.0)
            - torch.lgamma(y + theta + eps)
            - theta * torch.log(theta / (theta + mu + eps) + eps)
            - y * torch.log(mu / (theta + mu + eps) + eps)
        )


def coupled_gradient_loss(g, pos, grad):
    idx = torch.randperm(g.size(0))
    dg = torch.abs(g - g[idx])
    bd = torch.maximum(grad, grad[idx])
    w = torch.exp(-5.0 * bd)
    dp = torch.norm(pos - pos[idx], dim=1, keepdim=True)
    return (w * dg * (dp < 1.0)).mean()


def compute_metrics(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            mu, _, _, _ = model(
                b["vis"].to(DEVICE),
                b["pos"].to(DEVICE),
                b["grad"].to(DEVICE),
                b["library_size"].to(DEVICE),
            )
            preds.append(mu.cpu().numpy())
            targets.append(b["y"].numpy())

    p = np.log1p(np.concatenate(preds))
    t = np.log1p(np.concatenate(targets))
    mse = np.mean((p - t) ** 2)

    pcc, scc = [], []
    for i in range(p.shape[1]):
        if np.var(t[:, i]) > 1e-9:
            pcc.append(pearsonr(p[:, i], t[:, i])[0])
            scc.append(spearmanr(p[:, i], t[:, i])[0])

    return {"MSE": mse, "PCC": np.mean(pcc), "SCC": np.mean(scc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    svg_idx, _ = select_spatially_variable_genes(DATA_PATH)

    with h5py.File(DATA_PATH, "r") as f:
        sections = sorted([k for k in f.keys() if k.startswith("E")])

    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42)
    results = {"PCC": [], "SCC": [], "MSE": []}

    for fold, (tr, va) in enumerate(kf.split(sections)):
        train_sec = [sections[i] for i in tr]
        val_sec = [sections[i] for i in va]

        train_ds = MisarSectionDataset(DATA_PATH, svg_idx, train_sec)
        val_ds = MisarSectionDataset(DATA_PATH, svg_idx, val_sec)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=8)

        model = MoEST_Misar(NUM_SVGS).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

        nb_loss = NBLoss()
        align_loss = nn.MSELoss()

        best = -1.0
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            for b in train_loader:
                optimizer.zero_grad()
                mu, theta, g, z = model(
                    b["vis"].to(DEVICE),
                    b["pos"].to(DEVICE),
                    b["grad"].to(DEVICE),
                    b["library_size"].to(DEVICE),
                    apply_msm=True,
                )
                loss = (
                    LAMBDA_NB * nb_loss(b["y"].to(DEVICE), mu, theta)
                    + LAMBDA_ALIGN * align_loss(z, b["rna_z"].to(DEVICE))
                    + LAMBDA_GRAD * coupled_gradient_loss(g, b["pos"].to(DEVICE), b["grad"].to(DEVICE))
                )
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                metrics = compute_metrics(model, val_loader)
                if metrics["PCC"] > best:
                    best = metrics["PCC"]
                    patience = 0
                    torch.save(model.state_dict(), os.path.join(BASE_SAVE_DIR, f"fold{fold}.pth"))
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break

        metrics = compute_metrics(model, val_loader)
        for k in results:
            results[k].append(metrics[k])

    print("MSE:", np.mean(results["MSE"]), np.std(results["MSE"]))
    print("PCC:", np.mean(results["PCC"]), np.std(results["PCC"]))
    print("SCC:", np.mean(results["SCC"]), np.std(results["SCC"]))
