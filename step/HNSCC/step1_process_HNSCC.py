import scanpy as sc
import numpy as np
import h5py
import os
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

MASTER_FILE_PATH = "/path/to/input/MASTER_INDEX.h5ad"
DATA_DIR = "/path/to/input/SECTIONS_DIR"
OUTPUT_PATH = "/path/to/output/DATASET_RAWCOUNTS.h5"
PATCH_SIZE = 224

def compute_sobel_gradient(img_patch):
    if img_patch.size == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mn, mx = np.min(grad), np.max(grad)
    if mx - mn > 1e-8:
        grad = (grad - mn) / (mx - mn)
    else:
        grad = np.zeros_like(grad)
    return grad.astype(np.float32)

def force_unwrap(obj):
    while isinstance(obj, (np.ndarray, list)):
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            obj = obj.item()
        elif isinstance(obj, np.ndarray) and obj.size == 1:
            obj = obj[0]
        elif isinstance(obj, list) and len(obj) == 1:
            obj = obj[0]
        else:
            break
    return obj

def get_patches_and_gradients(img, coords, patch_size):
    coords = coords.astype(int)
    pad = patch_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    coords_padded = coords + pad

    patches = []
    gradients = []

    for x, y in coords_padded:
        patch = img_padded[y - pad:y + pad, x - pad:x + pad, :]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            if patch.size > 0:
                patch = cv2.resize(patch, (patch_size, patch_size))
            else:
                patch = np.zeros((patch_size, patch_size, 3), dtype=img.dtype)
        grad = compute_sobel_gradient(patch)
        patches.append(patch)
        gradients.append(grad)

    return np.array(patches), np.array(gradients)

if __name__ == "__main__":
    if not os.path.exists(MASTER_FILE_PATH):
        raise FileNotFoundError("Master index file not found")

    adata_3d = sc.read_h5ad(MASTER_FILE_PATH)

    raw_sections = np.unique(adata_3d.obs["n_section"].values)
    valid_sections = [int(x) for x in raw_sections if not np.isnan(x)]
    valid_sections.sort()

    if "raw" not in adata_3d.layers:
        raise ValueError("Layer 'raw' not found in AnnData")

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        gene_names = adata_3d.var_names.to_numpy().astype("S")
        h5f.create_dataset("gene_names", data=gene_names)

        for sec_num in tqdm(valid_sections, desc="Processing"):
            try:
                mask = (adata_3d.obs["n_section"] == sec_num)
                if np.sum(mask) == 0:
                    continue

                raw_sparse = adata_3d.layers["raw"][mask]
                subset_expr = raw_sparse.toarray() if hasattr(raw_sparse, "toarray") else np.array(raw_sparse)

                subset_coords_3d = adata_3d.obsm["spatial_3d_aligned"][mask]
                subset_coords_2d = adata_3d.obsm["spatial"][mask]

                target_suffix = f"_S{sec_num}."
                candidates = [
                    f for f in os.listdir(DATA_DIR)
                    if target_suffix in f and f.endswith(".h5ad")
                ]
                if not candidates:
                    continue

                raw_file_path = os.path.join(DATA_DIR, candidates[0])
                adata_raw = sc.read_h5ad(raw_file_path)

                spatial_dict = adata_raw.uns.get("spatial", {})
                img = None

                if isinstance(spatial_dict, dict) and "tissue" in spatial_dict:
                    img = force_unwrap(spatial_dict["tissue"])
                else:
                    if isinstance(spatial_dict, dict) and len(spatial_dict) > 0:
                        lib_key = list(spatial_dict.keys())[0]
                        lib = spatial_dict.get(lib_key, {})
                        if isinstance(lib, dict) and "tissue" in lib:
                            img = force_unwrap(lib["tissue"])
                        elif isinstance(lib, dict) and "images" in lib and "hires" in lib["images"]:
                            img = force_unwrap(lib["images"]["hires"])

                if img is None:
                    continue

                max_coord_x = np.max(subset_coords_2d[:, 0])
                img_w = img.shape[1]
                calculated_scale = img_w / max_coord_x if max_coord_x > 0 else 1.0
                final_scale = calculated_scale if calculated_scale < 0.8 else 1.0

                coords_to_crop = subset_coords_2d * final_scale
                patches, gradients = get_patches_and_gradients(img, coords_to_crop, PATCH_SIZE)

                grp = h5f.create_group(f"S{sec_num}")
                grp.create_dataset("patches", data=patches, compression="lzf")
                grp.create_dataset("sobel_gradients", data=gradients, compression="lzf")
                grp.create_dataset("coords_3d", data=subset_coords_3d)
                grp.create_dataset("expression", data=subset_expr, compression="lzf")

            except Exception:
                continue
