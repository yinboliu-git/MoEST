"""
Align MISAR slices using Spateo with UNI features

This script uses the Spateo package for spatial alignment, but instead of using
gene expression, we use UNI features as the input. This allows morphology-based
alignment of spatial transcriptomics slices.

Author: Anonymous
Date: 2026-01-29
"""

import os
import numpy as np
import h5py
import torch
import scanpy as sc
import spateo as st
import copy
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ================= Configuration =================
DATA_PATH = os.environ.get('DATA_PATH', './data/Misar/MISAR_3D_Final_RawCounts.h5')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './results/spateo_alignment')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Alignment parameters
CENTER_ALIGN = False         # Use sequential alignment (not center-based)
CENTER_SLICE = 0            # Center slice index (if CENTER_ALIGN=True)
ALLOW_FLIP = False          # Allow flipping during alignment
DISSIMILARITY = "cos"       # Distance metric: "cos" or "euclidean"
N_TOP_FEATURES = 2000       # Number of top UNI features to use

# Spateo parameters
SPATIAL_KEY = "spatial"
KEY_ADDED = "alignment_spatial"


# ================= Data Loading =================
def load_misar_slices_with_uni(h5_path: str,
                                selected_sections: Optional[List[str]] = None) -> List[sc.AnnData]:
    """
    Load MISAR slices from H5 file and create AnnData objects with UNI features

    Args:
        h5_path: Path to H5 file with UNI features
        selected_sections: List of section names to load (None = load all)

    Returns:
        List of AnnData objects, each with UNI features as .X
    """
    print(f"Loading MISAR data from: {h5_path}")

    adata_list = []

    with h5py.File(h5_path, 'r') as f:
        # Get all section names
        if selected_sections is None:
            sections = [k for k in f.keys()
                       if k[0].isalpha() and 'gene' not in k.lower()]
            sections = sorted(sections)
        else:
            sections = selected_sections

        print(f"Found {len(sections)} sections: {sections}")

        for sec in tqdm(sections, desc="Loading slices"):
            if sec not in f:
                print(f"Warning: Section {sec} not found, skipping")
                continue

            section = f[sec]

            # Check required fields
            if 'uni_features' not in section:
                print(f"Warning: {sec} missing UNI features, skipping")
                continue

            # Load data
            uni_features = section['uni_features'][:]  # (N, 1024)
            coords_3d = section['coords_3d'][:]        # (N, 3)

            # Load gene expression (for reference, not used in alignment)
            if 'expression' in section:
                expression = section['expression'][:]
            else:
                expression = None

            # Create AnnData with UNI features as .X
            # Key: Use UNI features instead of gene expression
            adata = sc.AnnData(
                X=uni_features,  # Use UNI features as "expression"
                obs={
                    'section': sec,
                    'x': coords_3d[:, 0],
                    'y': coords_3d[:, 1],
                    'z': coords_3d[:, 2]
               n            )

            # Add spatial coordinates
            adata.obsm[SPATIAL_KEY] = coords_3d[:, :2]  # Use XY for 2D alignment

            # Store original 3D coordinates
            adata.obsm['coords_3d'] = coords_3d

            # Store original gene expression if available
            if expression is not None:
                adata.obsm['gene_expression'] = expression

            # Add metadata
            adata.uns['section_name'] = sec
            adata.uns['n_spots'] = uni_features.shape[0]
            adata.uns['uni_dim'] = uni_features.shape[1]

            # Create fake variable names (UNI feature dimensions)
            adata.var_names = [f'UNI_{i}' for i in range(uni_features.shape[1])]

            adata_list.append(adata)
            print(f"  {sec}: {uni_features.shape[0]} spots, {uni_features.shape[1]} UNI features")

    print(f"\nLoaded {len(adata_list)} slices successfully")
    return adata_list


# ================= Spateo Alignment =================
def align_slices_with_spateo(
        adata_list: List[sc.AnnData],
        center_slice: int = 0,
        center_align: bool = False,
        spatial_key: str = "spatial",
        key_added: str = "alignment_spatial",
        device: str = "cpu",
        allow_flip: bool = False,
        dissimilarity: str = "cos",
        n_top_features: int = 2000
) -> List[sc.AnnData]:
    """
    Align slices using Spateo with UNI features

    This function wraps Spateo's morpho_align functions but uses UNI features
    instead of gene expression for alignment.

    Args:
        adata_list: List of AnnData objects with UNI features in .X
        center_slice: Index of center slice (for center_align=True)
        center_align: If True, align all to center; if False, sequential alignment
        spatial_key: Key for spatial coordinates in .obsm
        key_added: Key for aligned coordinates in .obsm
        device: 'cuda' or 'cpu'
        allow_flip: Allow flipping during alignment
        dissimilarity: Distance metric ('cos' or 'euclidean')
        n_top_features: Number of top UNI features to use

    Returns:
        List of aligned AnnData objects
    """
    print("\n" + "="*60)
    print("Spateo Alignment with UNI Features")
    print("="*60)
    print(f"Number of slices: {len(adata_list)}")
    print(f"Center align: {center_align}")
    print(f"Device: {device}")
    print(f"Dissimilarity: {dissimilarity}")
    print(f"Top features: {n_top_features}")

    # Deep copy to avoid modifying original data
    aligned_adata_list = copy.deepcopy(adata_list)

    # Preprocessing: Select top variable UNI features
    print("\nPreprocessing UNI features...")
    for i, adata in enumerate(aligned_adata_list):
        # Normalize UNI features (optional, UNI features are already normalized)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)

        # Select highly variable UNI features
        # This selects UNI dimensions with highest variance across spots
        if n_top_features < adata.shape[1]:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features)
            print(f"  Slice {i}: Selected {n_top_features}/{adata.shape[1]} UNI features")
        else:
            print(f"  Slice {i}: Using all {adata.shape[1]} UNI features")

    try:
        if center_align:
            # Center-based alignment: Align all slices to center slice
            print(f"\nPerforming center-based alignment (center={center_slice})...")

            aligned_slices, pi = st.align.morpho_align(
                models=aligned_adata_list,
                verbose=True,
                spatial_key=spatial_key,
                key_added=key_added,
                device=device,
            )

            print(f"Center alignment complete")

            # Apply alignment results
            for i, (src, dst) in enumerate(zip(aligned_adata_list, aligned_slices)):
                src.obsm[key_added] = dst.obsm[key_added]
                if i == center_slice:
                    # Center slice keeps original coordinates
                    src.obsm[spatial_key] = src.obsm[key_added].copy()

        else:
            # Sequential alignment: Align adjacent slices
            print("\nPerforming sequential alignment (adjacent slices)...")

            # Step 1: Compute transformation matrices
            print("  Computing transformation matrices...")
            transformation = st.align.morpho_align_transformation(
                models=aligned_adata_list,
                spatial_key=spatial_key,
                key_added=key_added,
                device=device,
                verbose=True,
                allow_flip=allow_flip,
                dissimilarity=dissimilarity,
            )

            print("  Transformation matrices computed")

            # Step 2: Apply transformations
            print("  Applying transformations...")
            aligned_slices = st.align.morpho_align_apply_transformation(
                models=aligned_adata_list,
                spatial_key=spatial_key,
                key_added=key_added,
                transformation=transformation,
            )

            print("  Transformations applied")
            aligned_adata_list = aligned_slices

    except Exception as e:
        print(f"\nError during alignment: {str(e)}")
        raise RuntimeError("Spateo alignment failed") from e

    # Clear GPU cache if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared")

    print("\n" + "="*60)
    print("Spateo alignment complete!")
    print("="*60)

    return aligned_adata_list


# ================= 3D Reconstruction =================
def reconstruct_3d_from_aligned(aligned_adata_list: List[sc.AnnData],
                                z_spacing: float = 1.0,
                                key_added: str = "alignment_spatial") -> sc.AnnData:
    """
    Reconstruct 3D volume from aligned 2D slices

    Args:
        aligned_adata_list: List of aligned AnnData objects
        z_spacing: Distance between slices in Z direction
        key_added: Key for aligned coordinates

    Returns:
        Combined AnnData with 3D coordinates
    """
    print("\nReconstructing 3D volume...")

    all_coords_3d = []
    all_uni_features = []
    all_sections = []

    for i, adata in enumerate(aligned_adata_list):
        # Get aligned XY coordinates
        aligned_xy = adata.obsm[key_added]

        # Create Z coordinates
        z_coords = np.full((aligned_xy.shape[0], 1), i * z_spacing)

        # Combine to 3D
        coords_3d = np.hstack([aligned_xy, z_coords])

        all_coords_3d.append(coords_3d)
        all_uni_features.append(adata.X)
        all_sections.extend([adata.uns['section_name']] * adata.shape[0])

    # Stack all data
    combined_coords = np.vstack(all_coords_3d)
    combined_uni = np.vstack(all_uni_features)

    # Create combined AnnData
    adata_3d = sc.AnnData(
        X=combined_uni,
        obs={'section': all_sections}
    )

    adata_3d.obsm['spatial_3d'] = combined_coords
    adata_3d.var_names = aligned_adata_list[0].var_names

    print(f"3D reconstruction complete: {adata_3d.shape[0]} spots, {adata_3d.shape[1]} features")

    return adata_3d


# ================= Visualization =================
def visualize_alignment(aligned_adata_list: List[sc.AnnData],
                       output_dir: str,
                       key_added: str = "alignment_spatial"):
    """
    Visualize aligned slices

    Args:
        aligned_adata_list: List of aligned AnnData objects
        output_dir: Directory to save plots
        key_added: Key for aligned coordinates
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # Plot 1: Overlay of all aligned slices
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(aligned_adata_list)))

    for i, adata in enumerate(aligned_adata_list):
        coords = adata.obsm[key_added]
        section_name = adata.uns['section_name']

        ax.scatter(coords[:, 0], coords[:, 1],
                  c=[colors[i]], s=1, alpha=0.5,
                  label=section_name)

    ax.set_xlabel('X (aligned)', fontsize=12)
    ax.set_ylabel('Y (aligned)', fontsize=12)
    ax.set_title('Spateo Alignment - All Slices Overlay', fontsize=14)
    ax.legend(markerscale=5, loc='best')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aligned_overlay.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/aligned_overlay.png")
    plt.close()

    # Plot 2: Individual slices in grid
    n_slices = len(aligned_adata_list)
    ncols = min(3, n_slices)
    nrows = (n_slices + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_slices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, adata in enumerate(aligned_adata_list):
        coords = adata.obsm[key_added]
        section_name = adata.uns['section_name']

        axes[i].scatter(coords[:, 0], coords[:, 1],
                       c=colors[i], s=1, alpha=0.6)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'{section_name}')
        axes[i].set_aspect('equal')

    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aligned_individual.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/aligned_individual.png")
    plt.close()

    # Plot 3: 3D reconstruction
    adata_3d = reconstruct_3d_from_aligned(aligned_adata_list, key_added=key_added)
    coords_3d = adata_3d.obsm['spatial_3d']
    sections = adata_3d.obs['section'].values

    fig = plt.figure(figsize=(18, 5))

    # XY view
    ax1 = fig.add_subplot(131)
    for i, sec in enumerate(np.unique(sections)):
        mask = sections == sec
        ax1.scatter(coords_3d[mask, 0], coords_3d[mask, 1],
                   c=[colors[i]], s=1, alpha=0.5, label=sec)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY View (Top)')
    ax1.legend(markerscale=5)
    ax1.set_aspect('equal')

    # XZ view
    ax2 = fig.add_subplot(132)
    for i, sec in enumerate(np.unique(sections)):
        mask = sections == sec
        ax2.scatter(coords_3d[mask, 0], coords_3d[mask, 2],
                   c=[colors[i]], s=1, alpha=0.5, label=sec)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('XZ View (Side)')

    # YZ view
    ax3 = fig.add_subplot(133)
    for i, sec in enumerate(np.unique(sections)):
        mask = sections == sec
        ax3.scatter(coords_3d[mask, 1], coords_3d[mask, 2],
                   c=[colors[i]], s=1, alpha=0.5, label=sec)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('YZ View (Front)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aligned_3d_views.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/aligned_3d_views.png")
    plt.close()


# ================= Main Pipeline =================
def main():
    """Main alignment pipeline"""

    print("="*60)
    print("MISAR Slice Alignment with Spateo + UNI Features")
    print("="*60)
    print(f"Data path: {DATA_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load MISAR slices with UNI features
    print("\n[Step 1] Loading MISAR slices...")

    # Option 1: Load all slices
    # adata_list = load_misar_slices_with_uni(DATA_PATH)

    # Option 2: Load specific slices (recommended for testing)
    selected_sections = ['E1', 'E2', 'E3', 'E4']  # Modify as needed
    adata_list = load_misar_slices_with_uni(DATA_PATH, selected_sections)

    if len(adata_list) < 2:
        raise ValueError("Need at least 2 slices for alignment")

    # Step 2: Align slices using Spateo
    print("\n[Step 2] Aligning slices with Spateo...")
    aligned_adata_list = align_slices_with_spateo(
        adata_list,
        center_slice=CENTER_SLICE,
        center_align=CENTER_ALIGN,
        spatial_key=SPATIAL_KEY,
        key_added=KEY_ADDED,
        device=DEVICE,
        allow_flip=ALLOW_FLIP,
        dissimilarity=DISSIMILARITY,
        n_top_features=N_TOP_FEATURES
    )

    # Step 3: Reconstruct 3D volume
    print("\n[Step 3] Reconstructing 3D volume...")
    adata_3d = reconstruct_3d_from_aligned(aligned_adata_list, key_added=KEY_ADDED)

    # Step 4: Visualize results
    print("\n[Step 4] Visualizing results...")
    visualize_alignment(aligned_adata_list, OUTPUT_DIR, key_added=KEY_ADDED)

    # Step 5: Save results
    print("\n[Step 5] Saving results...")

    # Save individual aligned slices
    for i, adata in enumerate(aligned_adata_list):
        section_name = adata.uns['section_name']
        output_file = f"{OUTPUT_DIR}/aligned_{section_name}.h5ad"
        adata.write_h5ad(output_file)
        print(f"Saved: {output_file}")

    # Save combined 3D data
    output_3d = f"{OUTPUT_DIR}/aligned_3d_combined.h5ad"
    adata_3d.write_h5ad(output_3d)
    print(f"Saved: {output_3d}")

    print("\n" + "="*60)
    print("Alignment pipeline complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

    return aligned_adata_list, adata_3d


if __name__ == "__main__":
    aligned_adata_list, adata_3d = main()
