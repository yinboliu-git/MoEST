import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ================= Configuration Area =================
# Use environment variable or default relative path
# 1. Data and model paths
H5_PATH = os.environ.get('H5_PATH', './data/her2st/HER2_3D_Final_RawCounts.h5')
MODEL_PATH = os.environ.get('MODEL_PATH', './checkpoints/her2_moest_plus_multi/best_model.pth')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './results/her2_vis')

# 2. Select a section for visualization (e.g., C1 or D1)
TARGET_SECTION = "D1"

# 3. Model parameters (must match training configuration)
NUM_HVG = 3000
DIM_UNI = 1024
DIM_HIDDEN = 256
NUM_EXPERTS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Model Definition (must match training code) =================
# For convenience, simplified copy of core classes, in production should import
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
        probs = F.softmax(logits, dim=-1) # Get probabilities for visualization
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)

        output = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_indices[:, i]
            val = topk_probs[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    output[mask] += val[mask] * self.experts[e](x[mask])
        return output, probs # Return probs for visualization

class MoEST_Plus_Inference(nn.Module):
    def __init__(self, num_genes, dim_uni=1024, dim_hidden=256):
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
            nn.Linear(dim_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, vis, pos, grad):
        z = self.img_enc(vis) + self.pos_enc(pos)
        z = self.spatial_ctx(z)
        z_moe, router_probs = self.moe(z, grad) # Get Router probabilities
        z = z + z_moe

        preds = self.gene_decoder(z).view(z.shape[0], -1, 2)
        mu = F.softplus(preds[:, :, 0])
        g = self.func_head(z)

        return mu, g, router_probs

# ================= Helper Plotting Functions =================
def plot_spatial(coords, values, title, save_path, cmap='viridis', s=10):
    plt.figure(figsize=(6, 6))
    # Flip Y axis to match image coordinate system
    sc = plt.scatter(coords[:, 0], -coords[:, 1], c=values, cmap=cmap, s=s, alpha=0.9)
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# ================= Main Program =================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load HVG indices (must match training)
    # Here we recalculate HVG indices, or better to load saved ones if available
    # For simplicity, we assume training used variance Top-3000
    print("Recalculating HVG indices for alignment...")
    with h5py.File(H5_PATH, 'r') as f:
        gene_names = f['gene_names'][:].astype(str)
        # Quick HVG calculation (use first 5 sections for estimation)
        keys = [k for k in list(f.keys())[:5] if k[0].isalpha()]
        all_expr = np.concatenate([f[k]['expression'][:] for k in keys], axis=0)
        variances = np.var(all_expr, axis=0)
        hvg_indices = np.argsort(variances)[-NUM_HVG:][::-1]
        final_genes = gene_names[hvg_indices]
        print(f"Locked {len(final_genes)} genes")

    # 2. Load model
    print(f"Loading model: {MODEL_PATH}")
    model = MoEST_Plus_Inference(num_genes=NUM_HVG).to(DEVICE)

    # Handle DataParallel saved weights (remove module. prefix)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False) # strict=False for minor differences
    model.eval()

    # 3. Read target section data
    print(f"Reading section: {TARGET_SECTION}")
    with h5py.File(H5_PATH, 'r') as f:
        if TARGET_SECTION not in f:
            raise ValueError(f"Section {TARGET_SECTION} does not exist!")

        vis = torch.tensor(f[TARGET_SECTION]['uni_features'][:]).float().to(DEVICE)
        pos = torch.tensor(f[TARGET_SECTION]['coords_3d'][:]).float().to(DEVICE)
        grad_map = f[TARGET_SECTION]['sobel_gradients'][:]
        grad_scalar = torch.tensor(np.mean(grad_map, axis=(1,2))).float().unsqueeze(1).to(DEVICE)

        # True expression (Raw Counts -> Log1p for visualization comparison)
        expr_raw = f[TARGET_SECTION]['expression'][:][:, hvg_indices]
        expr_true = np.log1p(expr_raw) # Visualization usually uses log1p

        # 2D coordinates for plotting (recover x, y from 3D coordinates)
        coords_plot = pos.cpu().numpy()[:, :2] # Take x, y

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        mu, g, router_probs = model(vis, pos, grad_scalar)

        # mu is NB mean, also do log1p for visualization
        expr_pred = np.log1p(mu.cpu().numpy())
        g_val = g.cpu().numpy().flatten()
        expert_assignments = router_probs.cpu().numpy() # (N, 4)

    # 5. Generate visualizations

    # --- A. Implicit functional gradient g ---
    print("Plotting functional gradient g...")

    plot_spatial(coords_plot, g_val, f"Functional Gradient (g)\n{TARGET_SECTION}",
                 f"{OUTPUT_DIR}/{TARGET_SECTION}_g_gradient.png", cmap='magma')

    # --- B. Expert routing map (MoE) ---
    print("Plotting expert routing distribution...")
    # Take expert ID with highest probability
    top_expert = np.argmax(expert_assignments, axis=1)
    plot_spatial(coords_plot, top_expert, f"MoE Expert Assignment\n{TARGET_SECTION}",
                 f"{OUTPUT_DIR}/{TARGET_SECTION}_experts.png", cmap='tab10', s=15)

    # --- C. Key gene comparison (FASN, ERBB2) ---
    # FASN and ERBB2 are key markers in HER2+ samples
    target_genes = ['FASN', 'ERBB2', 'CD24', 'ACTB']

    for gene in target_genes:
        if gene in final_genes:
            idx = np.where(final_genes == gene)[0][0]

            # True values
            plot_spatial(coords_plot, expr_true[:, idx], f"{gene} (True)",
                         f"{OUTPUT_DIR}/{TARGET_SECTION}_{gene}_true.png", cmap='viridis')

            # Predicted values
            plot_spatial(coords_plot, expr_pred[:, idx], f"{gene} (Pred)",
                         f"{OUTPUT_DIR}/{TARGET_SECTION}_{gene}_pred.png", cmap='viridis')

            # Calculate single gene PCC
            pcc, _ = pearsonr(expr_true[:, idx], expr_pred[:, idx])
            print(f"   {gene}: PCC = {pcc:.4f}")
        else:
            print(f"   Gene {gene} not in HVG list, skipping.")

    print("\nAll visualizations complete! Please check results/her2_vis folder.")
