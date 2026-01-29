import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr

# ================= Configuration Area =================
# Use environment variable or default relative path
DATA_PATH = os.environ.get('DATA_PATH', './data/her2st/HER2_3D_Final_RawCounts.h5')
SAVE_DIR = os.environ.get('SAVE_DIR', './checkpoints/her2_moest_plus_final')

# Training hyperparameters
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_HVG = 3000         # Only predict highly variable genes
LAMBDA_COUPLE = 0.1    # Coupling loss weight
MASK_RATIO = 0.15      # MSM masking ratio

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. Dataset (HVG Supported) =================
class HER2Dataset(Dataset):
    def __init__(self, h5_path, top_k_genes=3000):
        self.h5_path = h5_path
        self.indices = []

        print("Scanning data and filtering highly variable genes (HVG)...")
        all_exprs = []

        with h5py.File(h5_path, 'r') as f:
            self.gene_names = f['gene_names'][:].astype(str)
            keys = [k for k in f.keys() if k[0].isalpha() and 'gene' not in k and 'label' not in k]

            for k in tqdm(keys, desc="Indexing"):
                if 'uni_features' not in f[k]: continue
                n = f[k]['uni_features'].shape[0]
                for i in range(n):
                    self.indices.append((k, i))

                # Random sampling for variance calculation
                expr = f[k]['expression'][:]
                if len(expr) > 50:
                    idx = np.random.choice(len(expr), 50, replace=False)
                    all_exprs.append(expr[idx])
                else:
                    all_exprs.append(expr)

        # Calculate HVG
        concat_expr = np.concatenate(all_exprs, axis=0)
        variances = np.var(concat_expr, axis=0)
        self.gene_indices = np.argsort(variances)[-top_k_genes:][::-1]
        self.final_genes = self.gene_names[self.gene_indices]
        print(f"Locked {len(self.final_genes)} highly variable genes (Top-3000 HVG)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key, local_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            vis = f[key]['uni_features'][local_idx]
            pos = f[key]['coords_3d'][local_idx]
            grad_map = f[key]['sobel_gradients'][local_idx]
            grad_scalar = np.mean(grad_map) # Texture complexity scalar
            expr = f[key]['expression'][local_idx][self.gene_indices] # Only take HVG

        return {
            "vis": torch.tensor(vis, dtype=torch.float32),
            "pos": torch.tensor(pos, dtype=torch.float32),
            "grad": torch.tensor(grad_scalar, dtype=torch.float32).unsqueeze(0),
            "y": torch.tensor(expr, dtype=torch.float32)
        }

# ================= 2. Core Model Components =================

class FourierEncoding(nn.Module):
    """[3D Spatial] Fourier position encoding"""
    def __init__(self, input_dim=3, mapping_size=64, scale=10):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        self.out_dim = mapping_size * 2
    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialContextAttention(nn.Module):
    """[MSM] Spatial context attention module"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        # Treat Batch as Sequence, use neighboring points within batch for completion
        seq = x.unsqueeze(0)
        out, _ = self.attn(seq, seq, seq)
        return self.norm(x + out.squeeze(0))

class SparseMoELayer(nn.Module):
    """[MoE] Morphological gradient-guided mixture of experts layer"""
    def __init__(self, hidden_dim, num_experts=4, k=1):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        # Router input: Hidden + Sobel Gradient
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

# ================= 3. Complete Model Architecture =================

class MoEST_Plus(nn.Module):
    def __init__(self, num_genes, dim_uni=1024, dim_hidden=256):
        super().__init__()
        # Encoders
        self.img_enc = nn.Linear(dim_uni, dim_hidden)
        self.pos_enc = nn.Sequential(FourierEncoding(3), nn.Linear(128, dim_hidden))

        # MSM Context
        self.spatial_ctx = SpatialContextAttention(dim_hidden)

        # MoE
        self.moe = SparseMoELayer(dim_hidden)

        # Decoders
        # 1. Gene Expression Head (NB Params)
        self.gene_decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.LayerNorm(dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, num_genes * 2) # [mu, theta]
        )
        # 2. Functional Gradient Head (Implicit g)
        self.func_head = nn.Sequential(
            nn.Linear(dim_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, vis, pos, grad, apply_msm=False):
        # [Innovation 1] MSM: Random masking of visual features
        if apply_msm and self.training:
            mask = torch.rand(vis.shape[0], 1, device=vis.device) > MASK_RATIO
            vis = vis * mask

        # Encoding & Fusion
        z = self.img_enc(vis) + self.pos_enc(pos)

        # [Innovation 2] 3D Consistency via Context Attention
        z = self.spatial_ctx(z)

        # [Innovation 3] Morpho-guided MoE
        z = z + self.moe(z, grad)

        # Prediction
        # Gene Expression
        preds = self.gene_decoder(z).view(z.shape[0], -1, 2)
        mu = F.softplus(preds[:, :, 0])
        theta = F.softplus(preds[:, :, 1]) + 1e-6

        # Functional Gradient
        g = self.func_head(z)

        return mu, theta, g

# ================= 4. Loss Functions =================

def coupled_gradient_loss(g, pos, sobel_grad):
    """[Innovation 4] Dual gradient coupling loss"""
    # Random sampling of point pairs
    idx = torch.randperm(g.size(0))
    d_g = torch.abs(g - g[idx])

    # Physical boundary strength
    boundary = torch.maximum(sobel_grad, sobel_grad[idx])

    # Coupling logic: Strong boundary (large Sobel) -> Allow mutation (small weight)
    weight = torch.exp(-5.0 * boundary)

    # Only constrain local neighborhood
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

def compute_pcc(mu, y_true):
    y_pred = mu.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    pccs = []
    for i in range(y_pred.shape[1]):
        if np.var(y_true[:, i]) > 1e-6:
            c, _ = pearsonr(y_pred[:, i], y_true[:, i])
            if not np.isnan(c): pccs.append(c)
    return np.mean(pccs) if pccs else 0

# ================= 5. Main Training Loop =================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Data
    print("[Phase 1] Loading Dataset...")
    dataset = HER2Dataset(DATA_PATH, top_k_genes=NUM_HVG)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Build Model
    print("[Phase 2] Building MoEST-Plus Model...")
    model = MoEST_Plus(num_genes=NUM_HVG).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    nb_criterion = NBLoss()

    # 3. Train
    best_pcc = -1
    print("[Phase 3] Start Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            vis = batch['vis'].to(DEVICE)
            pos = batch['pos'].to(DEVICE)
            grad = batch['grad'].to(DEVICE)
            y = batch['y'].to(DEVICE)

            # Forward with MSM
            mu, theta, g = model(vis, pos, grad, apply_msm=True)

            # Calculate Losses
            loss_nb = nb_criterion(y, mu, theta)
            loss_couple = coupled_gradient_loss(g, pos, grad)

            # Combined Loss
            loss = loss_nb + LAMBDA_COUPLE * loss_couple

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(nb=loss_nb.item(), cpl=loss_couple.item())

        # Validation
        model.eval()
        val_pcc = 0
        with torch.no_grad():
            for batch in val_loader:
                vis = batch['vis'].to(DEVICE)
                pos = batch['pos'].to(DEVICE)
                grad = batch['grad'].to(DEVICE)
                y = batch['y'].to(DEVICE)

                # No MSM during inference
                mu, theta, g = model(vis, pos, grad, apply_msm=False)
                val_pcc += compute_pcc(mu, y)

        avg_pcc = val_pcc / len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val PCC: {avg_pcc:.4f}")

        if avg_pcc > best_pcc:
            best_pcc = avg_pcc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(f"SOTA Reached! Model Saved. (Best PCC: {best_pcc:.4f})")
