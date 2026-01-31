import scanpy as sc
import numpy as np
import h5py
import os
import cv2
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "/path/to/data/MISAR"
OUTPUT_H5 = os.path.join(BASE_DIR, "DATASET_XYT.h5")
PATCH_SIZE = 224

TIME_MAP = {
    "E11": 0.0, "E11_0": 0.0,
    "E13": 0.33, "E13_5": 0.33,
    "E15": 0.66, "E15_5": 0.66,
    "E18": 1.0, "E18_5": 1.0,
}

def get_patches_and_gradients(img, coords_pixel, patch_size):
    pad = patch_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    patches = []
    gradients = []

    for x, y in coords_pixel:
        x, y = int(x), int(y)
        x_pad, y_pad = x + pad, y + pad

        patch = img_padded[y_pad - pad:y_pad + pad, x_pad - pad:x_pad + pad, :]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

        patches.append(patch)
        gradients.append(mag.astype(np.float32))

    return np.array(patches), np.array(gradients)

if __name__ == "__main__":
    rna_files = sorted(glob.glob(os.path.join(BASE_DIR, "*_adata_rna_tissue_filtered.h5ad")))
    if len(rna_files) == 0:
        raise FileNotFoundError("No input .h5ad files found")

    with h5py.File(OUTPUT_H5, "w") as h5f:
        temp_ad = sc.read_h5ad(rna_files[0])
        h5f.create_dataset("gene_names", data=temp_ad.var_names.to_numpy().astype("S"))

        for rna_path in tqdm(rna_files):
            sample_name = None
            try:
                filename = os.path.basename(rna_path)
                sample_name = filename.split("_adata_rna")[0]

                img_name = f"{sample_name}-HE.jpg"
                candidates = [
                    os.path.join(BASE_DIR, img_name),
                    os.path.join(BASE_DIR, "HEimag", img_name),
                ]
                img_path = None
                for p in candidates:
                    if os.path.exists(p):
                        img_path = p
                        break
                if not img_path:
                    continue

                ad_rna = sc.read_h5ad(rna_path)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h_img, w_img = img.shape[:2]

                if "spatial" in ad_rna.obsm:
                    coords_raw = ad_rna.obsm["spatial"].copy()
                else:
                    coords_raw = ad_rna.obs[["x_coord", "y_coord"]].values.copy()

                x_raw = coords_raw[:, 0]
                y_raw = coords_raw[:, 1]

                x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min() + 1e-8)
                y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-8)

                x_pixel = (x_norm * 0.90 + 0.05) * w_img
                y_pixel = (y_norm * 0.90 + 0.05) * h_img
                coords_pixel = np.stack([x_pixel, y_pixel], axis=1)

                time_key = sample_name.split("-")[0]
                t_val = TIME_MAP.get(time_key, 0.5)
                t_vec = np.full_like(x_norm, t_val)

                coords_xyt = np.stack([x_norm, y_norm, t_vec], axis=1)

                patches, gradients = get_patches_and_gradients(img, coords_pixel, PATCH_SIZE)

                expr = ad_rna.X.toarray() if hasattr(ad_rna.X, "toarray") else ad_rna.X

                grp = h5f.create_group(sample_name)
                grp.create_dataset("patches", data=patches, compression="lzf")
                grp.create_dataset("sobel_gradients", data=gradients, compression="lzf")
                grp.create_dataset("coords_xyt", data=coords_xyt.astype(np.float32))
                grp.create_dataset("expression", data=expr.astype(np.float32), compression="lzf")

            except Exception:
                continue
