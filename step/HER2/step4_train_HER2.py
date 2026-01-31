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
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "/path/to/data/DATASET_PROCESSED.h5"
BASE_SAVE_DIR = "/path/to/output/checkpoints"
SVG_CACHE_PATH = "/path/to/cache/svg_cache.npz"

BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
PATIENCE = 15
NUM_SVGS = 2000

MIN_COUNTS_THRESHOLD = 20

LAMBDA_NB = 1.0
LAMBDA_ALIGN = 1.0
LAMBDA_GRAD = 0.1

DIM_UNI = 1024
DIM_HIDDEN = 256
SCVI_LATENT_DIM = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_SEED = 1234

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_patient_id(section_id: str) -> str:
    m = re.match(r"([A-Za-z]+)", section_id)
    return m.group(1) if m else section_id

def parse_section_number(section_id: str) -> int:
    m = re.search(r"(\d+)", section_id)
    return int(m.group(1)) if m else 0

def build_pseudo_z_map(all_sections):
    by_patient = {}
    for sid in all_sections:
        pid = parse_patient_id(sid)
        by_patient.setdefault(pid, []).append(sid)

    z_map = {}
    for pid, sids in by_patient.items():
        sids_sorted = sorted(sids, key=parse_section_number)
        n = len(sids_sorted)
        if n <= 1:
            z_map[sids_sorted[0]] = 0.0
        else:
            for r, sid in enumerate(sids_sorted):
                z_map[sid] = float(r) / float(n - 1)
    return z_map

def select_spatially_variable_genes_strict(h5_path, train_sections, n_top_genes=2000, seed=GLOBAL_SEED):
    if os.path.exists(SVG_CACHE_PATH):
        try:
            data = np.load(SVG_CACHE_PATH, allow_pickle=True)
            if "indices" in data and len(data["indices"]) > 0:
                return data["indices"], data["names"]
        except Exception:
            pass

    gene_counter = {}
    all_genes_raw = None
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, "r") as f:
        sample_keys = [k for k in f.keys() if k in train_sections]
        if not sample_keys:
            return np.arange(n_top_genes), []
        first_key = sample_keys[0]
        if "gene_names" in f:
            all_genes_raw = f["gene_names"][:].astype(str)
        else:
            all_genes_raw = f[first_key]["gene_names"][:].astype(str)

    subset_samples = list(train_sections)
    if len(subset_samples) > 10:
        subset_samples = list(rng.choice(subset_samples, size=10, replace=False))

    for sid in subset_samples:
        try:
            with h5py.File(h5_path, "r") as f:
                if sid not in f:
                    continue
                expr = f[sid]["expression"][:]
                if "coords_3d" in f[sid]:
                    coords = f[sid]["coords_3d"][:, :2]
                elif "spatial" in f[sid]:
                    coords = f[sid]["spatial"][:]
                else:
                    continue

            adata = ad.AnnData(X=expr)
            adata.var_names = all_genes_raw
            adata.obsm["spatial"] = coords

            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_THRESHOLD)
            if adata.n_obs < 10 or adata.n_vars < 50:
                continue

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            sq.gr.spatial_neighbors(adata, coord_type="generic")
            sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=10)

            res = adata.uns.get("moranI", None)
            if res is None or res.shape[0] == 0:
                continue
            sort_col = "I" if "I" in res.columns else ("moranI" if "moranI" in res.columns else None)
            if sort_col is None:
                continue

            top_genes = res.sort_values(sort_col, ascending=False).head(n_top_genes).index.tolist()
            for g in top_genes:
                gene_counter[g] = gene_counter.get(g, 0) + 1

        except Exception:
            continue

    if not gene_counter:
        final_indices = np.arange(n_top_genes)
        final_svgs = list(all_genes_raw[:n_top_genes]) if all_genes_raw is not None else []
    else:
        sorted_genes = sorted(gene_counter.items(), key=lambda x: x[1], reverse=True)
        final_svgs = [g[0] for g in sorted_genes[:n_top_genes]]
        gene_map = {name: i for i, name in enumerate(list(all_genes_raw))}
        gene_indices = [gene_map[g] for g in final_svgs if g in gene_map]
        final_indices = np.array(gene_indices, dtype=np.int64)

    os.makedirs(os.path.dirname(SVG_CACHE_PATH), exist_ok=True) if os.path.dirname(SVG_CACHE_PATH) else None
    np.savez(SVG_CACHE_PATH, indices=final_indices, names=final_svgs)
    return final_indices, final_svgs

class HER2Dataset_Ram(Dataset):
    def __init__(self, h5_path, allowed_sections, gene_indices, z_map):
        self.h5_path = h5_path
        self.gene_indices = gene_indices

        self.data_vis = []
        self.data_pos = []
        self.data_grad = []
        self.data_y = []
        self.data_lib_size = []
        self.data_rna_z = []

        with h5py.File(h5_path, "r") as f:
            for sid in tqdm(allowed_sections, desc="Loading", leave=False):
                if sid not in f:
                    continue
                grp = f[sid]
                if "expression" not in grp or not isinstance(grp["expression"], h5py.Dataset):
                    continue

                try:
                    expr_all = grp["expression"][:]
                    counts_per_spot = expr_all.sum(axis=1)

                    if "valid_mask" in grp:
                        valid_mask = grp["valid_mask"][:].astype(bool)
                    else:
                        valid_mask = counts_per_spot > MIN_COUNTS_THRESHOLD

                    if not np.any(valid_mask):
                        continue
                    if "uni_features" not in grp:
                        continue

                    vis_batch = grp["uni_features"][valid_mask]

                    if "coords_3d" in grp:
                        raw_coords = grp["coords_3d"][valid_mask]
                        if raw_coords.shape[1] == 2:
                            z_val = z_map.get(sid, 0.0)
                            z_col = np.full((raw_coords.shape[0], 1), z_val)
                            raw_coords = np.hstack([raw_coords, z_col])
                        else:
                            raw_coords[:, 2] = z_map.get(sid, 0.0)
                    else:
                        continue

                    xy = raw_coords[:, :2].astype(np.float32)
                    z = raw_coords[:, 2:].astype(np.float32)

                    xy_min = xy.min(axis=0)
                    xy_max = xy.max(axis=0)
                    xy_denom = xy_max - xy_min + 1e-6
                    xy_norm = (xy - xy_min) / xy_denom

                    pos_batch_norm = np.concatenate([xy_norm, z], axis=1)

                    if "sobel_gradients" in grp:
                        grad_batch = grp["sobel_gradients"][valid_mask]
                        if grad_batch.ndim > 1:
                            grad_batch = np.mean(grad_batch, axis=(1, 2))
                    else:
                        grad_batch = np.zeros(np.sum(valid_mask), dtype=np.float32)

                    self.data_vis.append(vis_batch)
                    self.data_pos.append(pos_batch_norm)
                    self.data_grad.append(grad_batch)
                    self.data_y.append(expr_all[valid_mask][:, self.gene_indices])
                    self.data_lib_size.append(counts_per_spot[valid_mask])

                    if "scvi_latent" in grp:
                        self.data_rna_z.append(grp["scvi_latent"][valid_mask])
                    else:
                        self.data_rna_z.append(np.zeros((np.sum(valid_mask), SCVI_LATENT_DIM), dtype=np.float32))

                except Exception:
                    continue

        if len(self.data_y) > 0:
            self.data_vis = torch.tensor(np.concatenate(self.data_vis), dtype=torch.float32)
            self.data_pos = torch.tensor(np.concatenate(self.data_pos), dtype=torch.float32)
            self.data_grad = torch.tensor(np.concatenate(self.data_grad), dtype=torch.float32).unsqueeze(1)
            self.data_y = torch.tensor(np.concatenate(self.data_y), dtype=torch.float32)
            self.data_lib_size = torch.tensor(np.concatenate(self.data_lib_size), dtype=torch.float32).unsqueeze(1)
            self.data_rna_z = torch.tensor(np.concatenate(self.data_rna_z), dtype=torch.float32)
        else:
            self.data_vis = torch.empty((0, DIM_UNI), dtype=torch.float32)
            self.data_pos = torch.empty((0, 3), dtype=torch.float32)
            self.data_grad = torch.empty((0, 1), dtype=torch.float32)
            self.data_y = torch.empty((0, len(gene_indices)), dtype=torch.float32)
            self.data_lib_size = torch.empty((0, 1), dtype=torch.float32)
            self.data_rna_z = torch.empty((0, SCVI_LATENT_DIM), dtype=torch.float32)

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
            ) for _ in range(num_experts)
        ])

    def forward(self, x, grad):
        router_input = torch.cat([x, grad], dim=-1)
        logits = self.router(router_input)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(self.top_k):
            idx = topk_indices[:, i]
            val = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    output[mask] += val[mask] * self.experts[e](x[mask])
        return output, probs

class MoEST_HER2(nn.Module):
    def __init__(self, num_genes, dim_vis=1024, dim_hidden=256, num_experts=4, dropout_rate=0.3):
        super().__init__()
        self.img_enc = nn.Linear(dim_vis, dim_hidden)
        self.pos_enc = nn.Sequential(
            LearnableFourierEncoding(input_dim=3, mapping_size=128, scale=10.0),
            nn.Linear(256, dim_hidden),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.spatial_ctx = nn.Identity()
        self.moe = SparseMoELayer(dim_hidden, dim_hidden, num_experts, dropout_rate=0.2)

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

        self.func_head = nn.Sequential(
            nn.Linear(dim_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, vis, pos, grad, library_size, apply_msm=False):
        if apply_msm and self.training:
            vis = vis * (torch.rand(vis.shape[0], 1, device=vis.device) > 0.2)

        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.dropout(z)
        z = self.spatial_ctx(z)
        z_moe, _ = self.moe(z, grad)
        z_final = z + z_moe

        preds = self.gene_decoder(z_final).view(z_final.shape[0], -1, 2)
        mu = F.softplus(preds[:, :, 0]) * library_size + 1e-6
        theta = F.softplus(preds[:, :, 1]) + 1e-6

        return mu, theta, self.func_head(z_final), self.align_projector(z_final)

class NBLoss(nn.Module):
    def forward(self, y_true, mu, theta):
        eps = 1e-8
        t1 = torch.lgamma(y_true + theta + eps)
        t2 = torch.lgamma(theta + eps)
        t3 = torch.lgamma(y_true + 1.0)
        t4 = theta * torch.log(theta / (theta + mu + eps) + eps)
        t5 = y_true * torch.log(mu / (theta + mu + eps) + eps)
        return torch.mean(t2 + t3 - t1 - t4 - t5)

def coupled_gradient_loss(g, pos, sobel_grad):
    idx = torch.randperm(g.size(0))
    d_g = torch.abs(g - g[idx])
    boundary = torch.maximum(sobel_grad, sobel_grad[idx])
    weight = torch.exp(-5.0 * boundary)
    d_pos = torch.norm(pos - pos[idx], dim=1, keepdim=True)
    mask = (d_pos < 1.0).float()
    return (weight * d_g * mask).mean()

def compute_comprehensive_metrics(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            vis = batch["vis"].to(device)
            pos = batch["pos"].to(device)
            grad = batch["grad"].to(device)
            y = batch["y"].to(device)
            lib = batch["library_size"].to(device)
            mu, _, _, _ = model(vis, pos, grad, lib)
            preds.append(mu.cpu().numpy())
            targets.append(y.cpu().numpy())

    if len(preds) == 0:
        return {"MSE": 999.0, "PCC": 0.0, "SCC": 0.0}

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    preds_log = np.log1p(preds)
    targets_log = np.log1p(targets)

    mse = np.mean((preds_log - targets_log) ** 2)
    pcc_list, scc_list = [], []
    num_genes = preds.shape[1]
    for g in range(num_genes):
        if np.var(targets_log[:, g]) > 1e-6:
            p, _ = pearsonr(preds_log[:, g], targets_log[:, g])
            if not np.isnan(p):
                pcc_list.append(p)
            s, _ = spearmanr(preds_log[:, g], targets_log[:, g])
            if not np.isnan(s):
                scc_list.append(s)

    return {
        "MSE": float(mse),
        "PCC": float(np.mean(pcc_list)) if pcc_list else 0.0,
        "SCC": float(np.mean(scc_list)) if scc_list else 0.0,
    }

def run_lopo_fold(fold_idx, test_patient_id, train_pool_sections, test_sections, all_sections):
    fold_seed = GLOBAL_SEED + fold_idx
    set_all_seeds(fold_seed)

    train_inner, val_inner = train_test_split(train_pool_sections, test_size=0.1, random_state=fold_seed)

    svg_indices, _ = select_spatially_variable_genes_strict(DATA_PATH, train_inner, n_top_genes=NUM_SVGS, seed=fold_seed)

    z_map = build_pseudo_z_map(all_sections)
    train_ds = HER2Dataset_Ram(DATA_PATH, train_inner, svg_indices, z_map)
    val_ds = HER2Dataset_Ram(DATA_PATH, val_inner, svg_indices, z_map)
    test_ds = HER2Dataset_Ram(DATA_PATH, test_sections, svg_indices, z_map)

    if len(test_ds) == 0 or len(train_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = MoEST_HER2(num_genes=NUM_SVGS, dim_vis=DIM_UNI, dim_hidden=DIM_HIDDEN, dropout_rate=DROPOUT_RATE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    nb_loss = NBLoss()
    align_loss = nn.MSELoss()

    save_dir = os.path.join(BASE_SAVE_DIR, f"patient_{test_patient_id}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_val_pcc = -1e9
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            vis = batch["vis"].to(DEVICE)
            pos = batch["pos"].to(DEVICE)
            grad = batch["grad"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            lib = batch["library_size"].to(DEVICE)
            z_teacher = batch["rna_z"].to(DEVICE)

            optimizer.zero_grad()
            mu, theta, g, z_student = model(vis, pos, grad, lib, apply_msm=True)
            loss = (
                LAMBDA_NB * nb_loss(y, mu, theta)
                + LAMBDA_ALIGN * align_loss(z_student, z_teacher)
                + LAMBDA_GRAD * coupled_gradient_loss(g, pos, grad)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_accum += loss.item()

        scheduler.step()

        val_metrics = compute_comprehensive_metrics(model, val_loader, DEVICE)
        val_pcc = val_metrics["PCC"]
        print(f"Fold {fold_idx} | Epoch {epoch+1:03d} | Loss {train_loss_accum/len(train_loader):.4f} | ValPCC {val_pcc:.4f}")

        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            patience_cnt = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    state = torch.load(best_model_path, map_location=DEVICE)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    test_metrics = compute_comprehensive_metrics(model, test_loader, DEVICE)
    print(f"Fold {fold_idx} | Patient {test_patient_id} | PCC {test_metrics['PCC']:.4f} | MSE {test_metrics['MSE']:.4f}")
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with h5py.File(DATA_PATH, "r") as f:
        keys = list(f.keys())
        all_samples = [
            k for k in keys
            if k and k[0].isalpha()
            and "gene" not in k
            and "label" not in k
            and "feature" not in k
        ]
    all_samples.sort()

    patient_ids = np.array([parse_patient_id(s) for s in all_samples], dtype=str)
    logo = LeaveOneGroupOut()

    results = {"MSE": [], "PCC": [], "SCC": []}

    for fold, (train_idx, test_idx) in enumerate(logo.split(all_samples, groups=patient_ids)):
        test_patient = patient_ids[test_idx[0]]
        train_pool = [all_samples[i] for i in train_idx]
        test_sections = [all_samples[i] for i in test_idx]

        metrics = run_lopo_fold(fold, test_patient, train_pool, test_sections, all_samples)
        if metrics:
            for k in results:
                results[k].append(metrics[k])

    if results["PCC"]:
        print(f"Avg PCC: {np.mean(results['PCC']):.4f} ± {np.std(results['PCC']):.4f}")
        print(f"Avg MSE: {np.mean(results['MSE']):.4f} ± {np.std(results['MSE']):.4f}")
        print(f"Avg SCC: {np.mean(results['SCC']):.4f} ± {np.std(results['SCC']):.4f}")
    else:
        print("No valid folds completed.")
