import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

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
import json
import scanpy as sc
import squidpy as sq
import anndata as ad
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

DATA_PATH = "data/MISAR_XYT_Processed.h5"
SVG_CACHE_PATH = "cache/svg_cache_moran.npz"
BASE_SAVE_DIR = "outputs/misar/checkpoints_moe_e4_topk2"

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
LAMBDA_IMAGE_RECON = 1.0
LAMBDA_MOE_BALANCE = 0.05
LAMBDA_ROUTER_ENTROPY = 0.01
LAMBDA_ROUTER_ZLOSS = 1e-3
MSM_MASK_RATIO = 0.2

DIM_UNI = 1024
DIM_HIDDEN = 256
SCVI_LATENT_DIM = 30
NUM_EXPERTS = 4
MOE_TOP_K = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_spatially_variable_genes(h5_path, n_top_genes=2000, cache_path="svg_cache_moran.npz"):

    if os.path.exists(cache_path):
        print(f"[Cache] Loading SVG cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        if len(data['indices']) > 0:
            print(f"Loaded {len(data['indices'])} genes")
            return data['indices'], data['names']
        else:
            print("SVG cache is empty. Recomputing.")

    print("[MISAR] Computing spatiotemporally variable genes with Moran's I.")
    gene_counter = {}
    all_genes_raw = None

    with h5py.File(h5_path, 'r') as f:
        samples = [k for k in f.keys() if k.startswith('E')]

        if 'gene_names' in f:
            all_genes_raw = f['gene_names'][:].astype(str)
        else:
            all_genes_raw = f[samples[0]]['gene_names'][:].astype(str)

    subset_samples = samples if len(samples) < 5 else random.sample(samples, 5)

    try:
        import squidpy as sq
    except ImportError:
        raise ImportError("squidpy is required to compute spatially variable genes.")

    for sid in tqdm(subset_samples, desc="Calculating SVGs"):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'expression' not in f[sid]: continue
                counts = f[sid]['expression'][:]

                coords = f[sid]['coords_xyt'][:, :2]

            adata = ad.AnnData(X=counts)
            adata.var_names = all_genes_raw
            adata.obsm['spatial'] = coords

            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_THRESHOLD)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            if adata.n_vars < 50: continue

            sq.gr.spatial_neighbors(adata, coord_type="generic")
            sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=10)

            res = adata.uns['moranI']
            sort_col = 'I' if 'I' in res.columns else 'moranI'
            if sort_col in res.columns:

                top_genes = res.sort_values(sort_col, ascending=False).head(n_top_genes).index.tolist()
                for g in top_genes:
                    gene_counter[g] = gene_counter.get(g, 0) + 1

        except Exception as e:
            print(f"Failed to process section {sid}: {e}")
            continue

    if not gene_counter:
        print("SVG selection failed. Falling back to the first genes.")
        final_indices = np.arange(n_top_genes)
        final_svgs = all_genes_raw[:n_top_genes]
        return final_indices, final_svgs

    sorted_genes = sorted(gene_counter.items(), key=lambda x: x[1], reverse=True)
    final_svgs = [g[0] for g in sorted_genes[:n_top_genes]]

    gene_indices = []
    gene_map = {name: i for i, name in enumerate(list(all_genes_raw))}

    for g in final_svgs:
        if g in gene_map:
            gene_indices.append(gene_map[g])

    final_indices = np.array(gene_indices)

    print(f"Saving SVG cache to {cache_path}")
    np.savez(cache_path, indices=final_indices, names=final_svgs)

    print(f"SVG selection completed: {len(final_indices)} genes")
    return final_indices, final_svgs

class MisarSectionDataset(Dataset):
    def __init__(self, h5_path, gene_indices, allowed_sections=None):
        self.h5_path = h5_path
        self.gene_indices = gene_indices

        print("[Dataset] Preloading data into RAM.")

        self.data_vis = []
        self.data_pos = []
        self.data_grad = []
        self.data_y = []
        self.data_lib_size = []
        self.data_rna_z = []

        with h5py.File(h5_path, 'r') as f:
            all_keys = [k for k in f.keys() if k.startswith('E')]
            if allowed_sections:
                keys_to_process = [k for k in all_keys if k in allowed_sections]
            else:
                keys_to_process = all_keys
            keys_to_process.sort()

            for k in tqdm(keys_to_process, desc="Loading Sections"):
                if 'uni_features' not in f[k]: continue

                expr_all = f[k]['expression'][:]
                counts_per_spot = expr_all.sum(axis=1)

                valid_mask = counts_per_spot > MIN_COUNTS_THRESHOLD
                if not np.any(valid_mask): continue

                vis_batch = f[k]['uni_features'][valid_mask]
                self.data_vis.append(vis_batch)

                raw_coords = f[k]['coords_xyt'][valid_mask]

                xy = raw_coords[:, :2].astype(np.float32)
                t = raw_coords[:, 2:].astype(np.float32)

                xy_min = xy.min(axis=0)
                xy_max = xy.max(axis=0)
                xy_denom = xy_max - xy_min + 1e-6

                xy_norm = (xy - xy_min) / xy_denom

                t_norm = t / 20.0

                pos_batch_norm = np.concatenate([xy_norm, t_norm], axis=1)
                self.data_pos.append(pos_batch_norm)

                if 'sobel_gradients' in f[k]:
                    grad_batch = f[k]['sobel_gradients'][valid_mask]
                    grad_batch = np.mean(grad_batch, axis=(1,2)) if grad_batch.ndim > 1 else grad_batch
                else:
                    grad_batch = np.zeros(np.sum(valid_mask))
                self.data_grad.append(grad_batch)

                y_batch = expr_all[valid_mask][:, self.gene_indices]
                self.data_y.append(y_batch)

                lib_batch = counts_per_spot[valid_mask]
                self.data_lib_size.append(lib_batch)

                if 'scvi_latent' in f[k]:
                    z_batch = f[k]['scvi_latent'][valid_mask]
                else:
                    z_batch = np.zeros((np.sum(valid_mask), SCVI_LATENT_DIM))
                self.data_rna_z.append(z_batch)

        self.data_vis = torch.tensor(np.concatenate(self.data_vis), dtype=torch.float32)

        self.data_pos = torch.tensor(np.concatenate(self.data_pos), dtype=torch.float32)
        self.data_grad = torch.tensor(np.concatenate(self.data_grad), dtype=torch.float32).unsqueeze(1)
        self.data_y = torch.tensor(np.concatenate(self.data_y), dtype=torch.float32)
        self.data_lib_size = torch.tensor(np.concatenate(self.data_lib_size), dtype=torch.float32).unsqueeze(1)
        self.data_rna_z = torch.tensor(np.concatenate(self.data_rna_z), dtype=torch.float32)

        print(f"Data loaded. Total spots: {len(self.data_y)}")

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return {
            "vis": self.data_vis[idx],
            "pos": self.data_pos[idx],
            "grad": self.data_grad[idx],
            "y": self.data_y[idx],
            "library_size": self.data_lib_size[idx],
            "rna_z": self.data_rna_z[idx]
        }

class LearnableFourierEncoding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=128, scale=10.0):
        super().__init__()
        self.output_dim = mapping_size * 2

        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale)
        self.scale = scale

    def forward(self, x):

        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SparseMoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2, dropout_rate=0.2):
        super(SparseMoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim + 1, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(), nn.Dropout(dropout_rate),
                nn.Linear(input_dim * 4, output_dim), nn.Dropout(dropout_rate)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, grad):
        router_input = torch.cat([x, grad], dim=-1)
        logits = self.router(router_input)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_probs_norm = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        batch_size = x.size(0)
        output = torch.zeros(batch_size, x.size(1), device=x.device)
        for i in range(self.top_k):
            idx = topk_indices[:, i]
            val = topk_probs_norm[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    output[mask] += val[mask] * self.experts[e](x[mask])
        return output, probs, topk_indices, logits

class MoEST_Misar(nn.Module):
    def __init__(self, num_genes, dim_vis=1024, dim_hidden=256, num_experts=4, top_k=2, dropout_rate=0.3, st_dim=3):
        super().__init__()
        self.img_enc = nn.Linear(dim_vis, dim_hidden)
        self.dim_vis = dim_vis

        self.pos_enc = nn.Sequential(
            LearnableFourierEncoding(input_dim=st_dim, mapping_size=128, scale=10.0),
            nn.Linear(256, dim_hidden),
            nn.GELU()
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.spatial_ctx = nn.Identity()

        self.moe = SparseMoELayer(dim_hidden, dim_hidden, num_experts=num_experts, top_k=top_k, dropout_rate=0.2)

        self.gene_decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, num_genes * 2)
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_vis),
        )
        self.align_projector = nn.Sequential(
            nn.Linear(dim_hidden, 128), nn.GELU(), nn.Linear(128, SCVI_LATENT_DIM)
        )
        self.func_head = nn.Sequential(
            nn.Linear(dim_hidden, 64), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, vis, pos, grad, library_size, apply_msm=False, msm_mask_ratio=MSM_MASK_RATIO):
        vis_target = vis
        msm_keep_mask = torch.ones(vis.shape[0], 1, device=vis.device, dtype=vis.dtype)
        if apply_msm and self.training:
             msm_keep_mask = (torch.rand(vis.shape[0], 1, device=vis.device) > msm_mask_ratio).to(vis.dtype)
             vis = vis * msm_keep_mask

        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.dropout(z)
        z = self.spatial_ctx(z)

        z_moe, router_probs, topk_indices, router_logits = self.moe(z, grad)
        z_final = z + z_moe

        preds = self.gene_decoder(z_final).view(z_final.shape[0], -1, 2)

        mu = F.softplus(preds[:, :, 0]) * library_size + 1e-6
        theta = F.softplus(preds[:, :, 1]) + 1e-6

        vis_recon = self.image_decoder(z_final)
        z_align_pred = self.align_projector(z_final)
        g = self.func_head(z_final)
        return (
            mu,
            theta,
            g,
            z_align_pred,
            router_probs,
            topk_indices,
            router_logits,
            vis_recon,
            vis_target,
            msm_keep_mask,
        )

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

def moe_load_balancing_loss(router_probs, topk_indices, num_experts):
    importance = router_probs.mean(dim=0)
    selected = F.one_hot(topk_indices, num_classes=num_experts).float().sum(dim=1)
    load = selected.mean(dim=0) / topk_indices.size(1)
    return num_experts * torch.sum(importance * load) - 1.0

def router_entropy_loss(router_probs):
    entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=1)
    max_entropy = np.log(router_probs.size(1))
    return 1.0 - entropy.mean() / max_entropy

def router_z_loss(router_logits):
    return torch.mean(torch.square(router_logits))

def compute_moe_aux_loss(router_probs, topk_indices, router_logits):
    balance = moe_load_balancing_loss(router_probs, topk_indices, NUM_EXPERTS)
    entropy = router_entropy_loss(router_probs)
    zloss = router_z_loss(router_logits)
    total = (
        LAMBDA_MOE_BALANCE * balance
        + LAMBDA_ROUTER_ENTROPY * entropy
        + LAMBDA_ROUTER_ZLOSS * zloss
    )
    return total, {
        "balance": balance.detach(),
        "entropy": entropy.detach(),
        "zloss": zloss.detach(),
    }

def masked_image_reconstruction_loss(vis_recon, vis_target, msm_keep_mask):
    masked = (msm_keep_mask.squeeze(-1) < 0.5)
    if masked.any():
        return F.mse_loss(vis_recon[masked], vis_target[masked])
    return F.mse_loss(vis_recon, vis_target) * 0.0

def format_routing_stats(router_probs, topk_indices):
    probs = router_probs.detach()
    indices = topk_indices.detach()
    soft = probs.mean(dim=0)
    hard = torch.bincount(torch.argmax(probs, dim=1), minlength=NUM_EXPERTS).float()
    hard = hard / hard.sum().clamp_min(1.0)
    selected = F.one_hot(indices, num_classes=NUM_EXPERTS).float().sum(dim=1)
    load = selected.mean(dim=0) / indices.size(1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean() / np.log(NUM_EXPERTS)
    max_prob = probs.max(dim=1).values.mean()
    fmt = lambda x: "[" + ", ".join(f"{v:.3f}" for v in x.detach().cpu().tolist()) + "]"
    return (
        f"soft={fmt(soft)} topk_load={fmt(load)} hard={fmt(hard)} "
        f"entropy={entropy.item():.3f} max_prob={max_prob.item():.3f}"
    )

def compute_routing_stats(model, loader, device, max_batches=4):
    model.eval()
    probs_all, topk_all = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            vis = batch['vis'].to(device)
            pos = batch['pos'].to(device)
            grad = batch['grad'].to(device)
            library_size = batch['library_size'].to(device)
            outputs = model(vis, pos, grad, library_size)
            probs_all.append(outputs[4])
            topk_all.append(outputs[5])
    if not probs_all:
        return "no routing stats"
    return format_routing_stats(torch.cat(probs_all, dim=0), torch.cat(topk_all, dim=0))

def compute_metrics(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            vis = batch['vis'].to(device)
            pos = batch['pos'].to(device)
            grad = batch['grad'].to(device)
            y = batch['y'].to(device)
            library_size = batch['library_size'].to(device)
            mu = model(vis, pos, grad, library_size)[0]
            preds.append(mu.cpu().numpy())
            targets.append(y.cpu().numpy())

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    p_log = np.log1p(p)
    t_log = np.log1p(t)
    mse = np.mean((p_log - t_log)**2)

    pccs, sccs = [], []
    num_genes = p.shape[1]
    for i in range(num_genes):
        if np.var(t_log[:, i]) > 1e-9:
            val, _ = pearsonr(p_log[:, i], t_log[:, i])
            if not np.isnan(val): pccs.append(val)
            s_val, _ = spearmanr(p_log[:, i], t_log[:, i])
            if not np.isnan(s_val): sccs.append(s_val)

    return {"PCC": np.mean(pccs), "SCC": np.mean(sccs), "MSE": mse}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-experts', type=int, default=NUM_EXPERTS)
    parser.add_argument('--top-k', type=int, default=MOE_TOP_K)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--num-folds', type=int, default=NUM_FOLDS)
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--validate-every', type=int, default=20)
    parser.add_argument('--data-path', type=str, default=DATA_PATH)
    parser.add_argument('--svg-cache-path', type=str, default=SVG_CACHE_PATH)
    parser.add_argument('--save-dir', type=str, default=BASE_SAVE_DIR)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--lambda-moe-balance', type=float, default=LAMBDA_MOE_BALANCE)
    parser.add_argument('--lambda-router-entropy', type=float, default=LAMBDA_ROUTER_ENTROPY)
    parser.add_argument('--lambda-router-zloss', type=float, default=LAMBDA_ROUTER_ZLOSS)
    parser.add_argument('--lambda-image-recon', type=float, default=LAMBDA_IMAGE_RECON)
    parser.add_argument('--msm-mask-ratio', type=float, default=MSM_MASK_RATIO)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > args.num_experts:
        raise ValueError(f"--top-k must be in [1, num_experts], got top_k={args.top_k}, num_experts={args.num_experts}")
    if args.num_folds < 2:
        raise ValueError(f"--num-folds must be >= 2, got {args.num_folds}")
    if args.validate_every < 1:
        raise ValueError(f"--validate-every must be >= 1, got {args.validate_every}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    NUM_EXPERTS = args.num_experts
    MOE_TOP_K = args.top_k
    EPOCHS = args.epochs
    NUM_FOLDS = args.num_folds
    BATCH_SIZE = args.batch_size
    DATA_PATH = args.data_path
    SVG_CACHE_PATH = args.svg_cache_path
    BASE_SAVE_DIR = args.save_dir
    LAMBDA_MOE_BALANCE = args.lambda_moe_balance
    LAMBDA_ROUTER_ENTROPY = args.lambda_router_entropy
    LAMBDA_ROUTER_ZLOSS = args.lambda_router_zloss
    LAMBDA_IMAGE_RECON = args.lambda_image_recon
    MSM_MASK_RATIO = args.msm_mask_ratio

    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    print(f"Starting MISAR Balanced-MoE training on GPU {args.gpu}")
    print(
        f"   MoE: num_experts={NUM_EXPERTS}, top_k={MOE_TOP_K}, "
        f"epochs={EPOCHS}, num_folds={NUM_FOLDS}, max_folds={args.max_folds}, "
        f"msm_mask_ratio={MSM_MASK_RATIO}, lambda_image_recon={LAMBDA_IMAGE_RECON}, "
        f"lambda_balance={LAMBDA_MOE_BALANCE}, "
        f"lambda_entropy={LAMBDA_ROUTER_ENTROPY}, "
        f"lambda_zloss={LAMBDA_ROUTER_ZLOSS}"
    )

    svg_indices, svg_names = select_spatially_variable_genes(
        DATA_PATH, n_top_genes=NUM_SVGS, cache_path=SVG_CACHE_PATH
    )

    with h5py.File(DATA_PATH, 'r') as f:
        all_sections = [k for k in f.keys() if k.startswith('E')]
    all_sections.sort()

    print(f"Found {len(all_sections)} sections for validation.")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    section_indices = np.arange(len(all_sections))

    fold_results_pcc = []
    fold_results_mse = []
    fold_results_scc = []

    for fold, (train_sec_idx, val_sec_idx) in enumerate(kf.split(section_indices)):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        print(f"\nFold {fold+1}/{NUM_FOLDS}")

        train_sections = [all_sections[i] for i in train_sec_idx]
        val_sections = [all_sections[i] for i in val_sec_idx]

        print(f"   Train Sections ({len(train_sections)}): {train_sections[:2]} ...")
        print(f"   Val Sections   ({len(val_sections)}): {val_sections}")

        train_dataset = MisarSectionDataset(DATA_PATH, svg_indices, allowed_sections=train_sections)
        val_dataset = MisarSectionDataset(DATA_PATH, svg_indices, allowed_sections=val_sections)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

        model = MoEST_Misar(
            num_genes=NUM_SVGS,
            dim_vis=DIM_UNI,
            dim_hidden=DIM_HIDDEN,
            num_experts=NUM_EXPERTS,
            top_k=MOE_TOP_K,
        )
        if not args.no_compile:
            model = torch.compile(model)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        nb_loss = NBLoss()
        align_loss = nn.MSELoss()

        best_metrics = {"PCC": -1.0, "SCC": 0.0, "MSE": 999.0}
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_main_loss = 0
            train_img_recon_loss = 0
            train_moe_loss = 0
            train_balance_loss = 0
            train_entropy_loss = 0
            train_zloss = 0
            last_router_probs = None
            last_topk_indices = None

            for batch in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
                vis = batch['vis'].to(DEVICE)
                pos = batch['pos'].to(DEVICE)
                grad = batch['grad'].to(DEVICE)
                y = batch['y'].to(DEVICE)
                z_teacher = batch['rna_z'].to(DEVICE)

                optimizer.zero_grad()
                library_size = batch['library_size'].to(DEVICE)

                (
                    mu,
                    theta,
                    g,
                    z_student,
                    router_probs,
                    topk_indices,
                    router_logits,
                    vis_recon,
                    vis_target,
                    msm_keep_mask,
                ) = model(
                    vis,
                    pos,
                    grad,
                    library_size,
                    apply_msm=True,
                    msm_mask_ratio=MSM_MASK_RATIO,
                )

                main_loss = LAMBDA_NB * nb_loss(y, mu, theta) +\
                            LAMBDA_ALIGN * align_loss(z_student, z_teacher) +\
                            LAMBDA_GRAD * coupled_gradient_loss(g, pos, grad)
                image_recon_loss = masked_image_reconstruction_loss(
                    vis_recon, vis_target, msm_keep_mask
                )
                moe_aux_loss, moe_terms = compute_moe_aux_loss(
                    router_probs, topk_indices, router_logits
                )
                loss = main_loss + LAMBDA_IMAGE_RECON * image_recon_loss + moe_aux_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                train_main_loss += main_loss.item()
                train_img_recon_loss += image_recon_loss.item()
                train_moe_loss += moe_aux_loss.item()
                train_balance_loss += moe_terms["balance"].item()
                train_entropy_loss += moe_terms["entropy"].item()
                train_zloss += moe_terms["zloss"].item()
                last_router_probs = router_probs
                last_topk_indices = topk_indices

            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            avg_main_loss = train_main_loss / len(train_loader)
            avg_img_recon_loss = train_img_recon_loss / len(train_loader)
            avg_moe_loss = train_moe_loss / len(train_loader)
            avg_balance_loss = train_balance_loss / len(train_loader)
            avg_entropy_loss = train_entropy_loss / len(train_loader)
            avg_zloss = train_zloss / len(train_loader)
            train_routing = (
                format_routing_stats(last_router_probs, last_topk_indices)
                if last_router_probs is not None else "no routing stats"
            )

            if (epoch + 1) % args.validate_every == 0:

                metrics = compute_metrics(model, val_loader, DEVICE)
                val_pcc = metrics['PCC']
                val_routing = compute_routing_stats(model, val_loader, DEVICE)
                print(
                    f"   Ep {epoch+1:02d} | Loss: {avg_train_loss:.4f} "
                    f"(main={avg_main_loss:.4f}, img_recon={avg_img_recon_loss:.4f}, "
                    f"moe={avg_moe_loss:.4f}) | "
                    f"bal={avg_balance_loss:.4f}, ent_pen={avg_entropy_loss:.4f}, "
                    f"z={avg_zloss:.4f} | PCC: {val_pcc:.4f} (Validated)"
                )
                print(f"      train routing: {train_routing}")
                print(f"      val routing:   {val_routing}")

                if val_pcc > best_metrics["PCC"]:
                    best_metrics = metrics
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(BASE_SAVE_DIR, f"fold{fold}_best.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print("Early stopping")
                        break
            else:

                print(
                    f"   Ep {epoch+1:02d} | Loss: {avg_train_loss:.4f} "
                    f"(main={avg_main_loss:.4f}, img_recon={avg_img_recon_loss:.4f}, "
                    f"moe={avg_moe_loss:.4f}) | "
                    f"bal={avg_balance_loss:.4f}, ent_pen={avg_entropy_loss:.4f}, "
                    f"z={avg_zloss:.4f} | PCC: ---- (Skipped)"
                )
                print(f"      train routing: {train_routing}")

        print(f"Fold {fold+1} final metrics: {best_metrics}")
        fold_results_pcc.append(best_metrics["PCC"])
        fold_results_mse.append(best_metrics["MSE"])
        fold_results_scc.append(best_metrics["SCC"])

    print("\n" + "="*60)
    print("MISAR final results")
    print("="*60)
    print(f"Avg MSE: {np.mean(fold_results_mse):.4f} +/- {np.std(fold_results_mse):.4f}")
    print(f"Avg PCC: {np.mean(fold_results_pcc):.4f} +/- {np.std(fold_results_pcc):.4f}")
    print(f"Avg SCC: {np.mean(fold_results_scc):.4f} +/- {np.std(fold_results_scc):.4f}")
    summary = {
        "num_experts": NUM_EXPERTS,
        "top_k": MOE_TOP_K,
        "epochs": EPOCHS,
        "num_folds": NUM_FOLDS,
        "max_folds": args.max_folds,
        "batch_size": BATCH_SIZE,
        "validate_every": args.validate_every,
        "lambda_moe_balance": LAMBDA_MOE_BALANCE,
        "lambda_router_entropy": LAMBDA_ROUTER_ENTROPY,
        "lambda_router_zloss": LAMBDA_ROUTER_ZLOSS,
        "lambda_image_recon": LAMBDA_IMAGE_RECON,
        "msm_mask_ratio": MSM_MASK_RATIO,
        "fold_pcc": [float(x) for x in fold_results_pcc],
        "fold_mse": [float(x) for x in fold_results_mse],
        "fold_scc": [float(x) for x in fold_results_scc],
        "avg_pcc": float(np.mean(fold_results_pcc)) if fold_results_pcc else None,
        "std_pcc": float(np.std(fold_results_pcc)) if fold_results_pcc else None,
        "avg_mse": float(np.mean(fold_results_mse)) if fold_results_mse else None,
        "std_mse": float(np.std(fold_results_mse)) if fold_results_mse else None,
        "avg_scc": float(np.mean(fold_results_scc)) if fold_results_scc else None,
        "std_scc": float(np.std(fold_results_scc)) if fold_results_scc else None,
    }
    summary_path = os.path.join(BASE_SAVE_DIR, "summary_balanced_moe.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
