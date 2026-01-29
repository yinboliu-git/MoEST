import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr
import random
import argparse

# ================= Configuration Area =================
# Use environment variable or default relative path
DATA_PATH = os.environ.get('DATA_PATH', './data/her2st/HER2_3D_Final_RawCounts.h5')
BASE_SAVE_DIR = "checkpoints_version2/her2_kfold_experiment"

# Experiment parameters
NUM_FOLDS = 5          # 5-fold cross validation
TRAIN_RATIO = 0.5      # 50% slices for training
BATCH_SIZE = 1024      # Recommended 1024 or 2048 for A100/H100
EPOCHS = 80            # 80 epochs per fold
LEARNING_RATE = 2e-4   # Initial learning rate
NUM_HVG = 1000

# Model parameters (increased Hidden Dim for better capacity)
DIM_UNI = 1024
DIM_HIDDEN = 512       # Increased from 256 to 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 1. Dataset with Section Filtering Support =================
class HER2Dataset_Split(Dataset):
    def __init__(self, h5_path, allowed_sections, gene_indices=None):
        """
        allowed_sections: List of allowed sections (e.g., ['A1', 'B1'])
        gene_indices: Specified gene indices (to keep train/test genes consistent)
        """
        self.h5_path = h5_path
        self.indices = []
        self.allowed_sections = allowed_sections

        with h5py.File(h5_path, 'r') as f:
            # 1. If gene_indices not provided, calculate HVG
            if gene_indices is None:
                all_exprs = []
                # To calculate HVG, scan all allowed_sections
                for k in allowed_sections:
                    if k in f and 'expression' in f[k]:
                        # Sample for acceleration
                        expr = f[k]['expression'][:]
                        if len(expr) > 100:
                            idx = np.random.choice(len(expr), 100, replace=False)
                            all_exprs.append(expr[idx])
                        else:
                            all_exprs.append(expr)

                concat_expr = np.concatenate(all_exprs, axis=0)
                variances = np.var(concat_expr, axis=0)
                self.gene_indices = np.argsort(variances)[-NUM_HVG:][::-1]

                # Get gene names
                full_gene_names = f['gene_names'][:].astype(str)
                self.final_genes = full_gene_names[self.gene_indices]
            else:
                self.gene_indices = gene_indices

            # 2. Build indices
            for k in allowed_sections:
                if k not in f: continue
                if 'uni_features' not in f[k]: continue

                n = f[k]['uni_features'].shape[0]
                for i in range(n):
                    self.indices.append((k, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key, local_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            vis = f[key]['uni_features'][local_idx]
            pos = f[key]['coords_3d'][local_idx]
            grad_map = f[key]['sobel_gradients'][local_idx]
            grad_scalar = np.mean(grad_map)
            # Only take selected HVG
            expr = f[key]['expression'][local_idx][self.gene_indices]

        return {
            "vis": torch.tensor(vis, dtype=torch.float32),
            "pos": torch.tensor(pos, dtype=torch.float32),
            "grad": torch.tensor(grad_scalar, dtype=torch.float32).unsqueeze(0),
            "y": torch.tensor(expr, dtype=torch.float32)
        }

# ================= 2. Model Components (MoEST Plus) =================

class FourierEncoding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=64, scale=10):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialContextAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        seq = x.unsqueeze(0)
        out, _ = self.attn(seq, seq, seq)
        return self.norm(x + out.squeeze(0))

class SparseMoELayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, k=1):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim + 1, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_experts)
        ])
    def forward(self, x, grad):
        router_in = torch.cat([x, grad], dim=-1)
        logits = self.router(router_in)
        topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), self.k, dim=-1)
        output = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_indices[:, i]
            val = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    output[mask] += val[mask] * self.experts[e](x[mask])
        return output

class MoEST_Plus(nn.Module):
    def __init__(self, num_genes, dim_uni=1024, dim_hidden=512):
        super().__init__()
        self.img_enc = nn.Linear(dim_uni, dim_hidden)
        self.pos_enc = nn.Sequential(FourierEncoding(3), nn.Linear(128, dim_hidden))
        self.spatial_ctx = SpatialContextAttention(dim_hidden)
        self.moe = SparseMoELayer(dim_hidden)
        self.gene_decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, num_genes * 2)
        )
        self.func_head = nn.Sequential(
            nn.Linear(dim_hidden, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, vis, pos, grad, apply_msm=False):
        if apply_msm and self.training:
            mask = torch.rand(vis.shape[0], 1, device=vis.device) > 0.15
            vis = vis * mask
        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.spatial_ctx(z)
        z = z + self.moe(z, grad)
        preds = self.gene_decoder(z).view(z.shape[0], -1, 2)
        mu = F.softplus(preds[:, :, 0])
        theta = F.softplus(preds[:, :, 1]) + 1e-6
        g = self.func_head(z)
        return mu, theta, g

def coupled_gradient_loss(g, pos, sobel_grad):
    idx = torch.randperm(g.size(0))
    d_g = torch.abs(g - g[idx])
    boundary = torch.maximum(sobel_grad, sobel_grad[idx])
    weight = torch.exp(-5.0 * boundary)
    d_pos = torch.norm(pos - pos[idx], dim=1, keepdim=True)
    mask = (d_pos < 1.0).float()
    return (weight * d_g * mask).mean()

class NBLoss(nn.Module):
    def forward(self, y_true, mu, theta):
        eps = 1e-8
        t1 = torch.lgamma(y_true + theta + eps)
        t2 = torch.lgamma(theta + eps)
        t3 = torch.lgamma(y_true + 1.0)
        t4 = theta * torch.log(theta / (theta + mu + eps) + eps)
        t5 = y_true * torch.log(mu / (theta + mu + eps) + eps)
        return torch.mean(t2 + t3 - t1 - t4 - t5)

# ================= 3. Key Improvement: Log1p PCC Calculation =================
def compute_pcc_log1p(mu, y_true):
    """
    Key fix: Calculate PCC in log1p space
    Raw counts PCC is easily affected by outliers, log1p is the standard practice in SOTA papers
    """
    # Predicted mu is in raw scale, convert to log1p
    y_pred = torch.log1p(mu).detach().cpu().numpy()
    # True y_true is also in raw scale, convert to log1p
    y_true = torch.log1p(y_true).detach().cpu().numpy()

    pccs = []
    for i in range(y_pred.shape[1]):
        if np.var(y_true[:, i]) > 1e-6:
            c, _ = pearsonr(y_pred[:, i], y_true[:, i])
            if not np.isnan(c): pccs.append(c)
    return np.mean(pccs) if pccs else 0

# ================= 4. Five-Fold Experiment Main Loop =================
def run_fold(fold_idx, train_sections, test_sections):
    print(f"\n[Fold {fold_idx+1}/{NUM_FOLDS}] Starting...")
    print(f"   Train Sections ({len(train_sections)}): {train_sections}")
    print(f"   Test Sections  ({len(test_sections)}): {test_sections}")

    save_dir = os.path.join(BASE_SAVE_DIR, f"fold_{fold_idx+1}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Build datasets
    # Training set calculates HVG
    train_dataset = HER2Dataset_Split(DATA_PATH, allowed_sections=train_sections)
    # Test set must use the same genes as training set (inherit gene_indices)
    test_dataset = HER2Dataset_Split(DATA_PATH, allowed_sections=test_sections,
                                     gene_indices=train_dataset.gene_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model
    model = MoEST_Plus(num_genes=NUM_HVG, dim_hidden=DIM_HIDDEN)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Learning rate scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    nb_criterion = NBLoss()

    best_pcc = -1

    # 3. Training
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            vis = batch['vis'].to(DEVICE)
            pos = batch['pos'].to(DEVICE)
            grad = batch['grad'].to(DEVICE)
            y = batch['y'].to(DEVICE)

            mu, theta, g = model(vis, pos, grad, apply_msm=True)
            loss = nb_criterion(y, mu, theta) + 0.1 * coupled_gradient_loss(g, pos, grad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_pcc = 0
            with torch.no_grad():
                for batch in test_loader:
                    vis = batch['vis'].to(DEVICE)
                    pos = batch['pos'].to(DEVICE)
                    grad = batch['grad'].to(DEVICE)
                    y = batch['y'].to(DEVICE)

                    mu, theta, g = model(vis, pos, grad, apply_msm=False)
                    # Use optimized PCC calculation
                    val_pcc += compute_pcc_log1p(mu, y)

            avg_pcc = val_pcc / len(test_loader)
            print(f"   Epoch {epoch+1} | Val PCC (Log1p): {avg_pcc:.4f}")

            if avg_pcc > best_pcc:
                best_pcc = avg_pcc
                if isinstance(model, nn.DataParallel):
                    state = model.module.state_dict()
                else:
                    state = model.state_dict()
                torch.save(state, os.path.join(save_dir, "best_model.pth"))

    print(f"Fold {fold_idx+1} completed! Best PCC: {best_pcc:.4f}")
    return best_pcc

if __name__ == "__main__":
    # ================= Modified section: Parameter control =================
    parser = argparse.ArgumentParser(description='HER2 5-Fold Training Version 2')
    parser.add_argument('--fold', type=int, default=0, help='Specify which fold to run (0-4)')
    parser.add_argument('--gpu', type=str, default='0', help='Specify visible GPU')
    args = parser.parse_args()

    # 1. Basic configuration
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # 2. Read all sections and sort (ensure all tasks see the same order)
    print("Reading section list...")
    with h5py.File(DATA_PATH, 'r') as f:
        all_keys = [k for k in f.keys() if k[0].isalpha() and 'gene' not in k]
    all_keys.sort()

    # 3. Determine current fold split (using fixed seed)
    seed = 42  # Global fixed seed to ensure consistent splits
    random.seed(seed)
    np.random.seed(seed)

    shuffled_keys = all_keys[:]
    random.shuffle(shuffled_keys)

    # Simple K-Fold split logic
    total_samples = len(shuffled_keys)
    fold_size = total_samples // NUM_FOLDS

    # Calculate current fold index range
    start_idx = args.fold * fold_size
    end_idx = (args.fold + 1) * fold_size if args.fold < NUM_FOLDS - 1 else total_samples

    # Split test set and training set
    test_sec = shuffled_keys[start_idx:end_idx]
    train_sec = [k for k in shuffled_keys if k not in test_sec]

    print(f"\nStarting task: Fold {args.fold}")
    print(f"Assigned GPU: {args.gpu}")
    print(f"   Train Set: {len(train_sec)} slices")
    print(f"   Test Set : {len(test_sec)} slices")

    # 4. Run single fold task
    run_fold(args.fold, train_sec, test_sec)
