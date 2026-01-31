import os
import pandas as pd
import numpy as np
import h5py
import cv2
from tqdm import tqdm
import re
import glob

BASE_DIR = "/path/to/data/root"
OUTPUT_PATH = "/path/to/output/DATASET.h5"
PATCH_SIZE = 224

LABEL_MAP = {
    'invasive cancer': 1,
    'breast glands': 2,
    'immune infiltrate': 3,
    'cancer in situ': 4,
    'connective tissue': 5,
    'adipose tissue': 6,
    'undetermined': 0,
    'Unlabeled': -1
}

def compute_sobel_gradient(img_patch):
    if img_patch.size == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mn, mx = grad.min(), grad.max()
    if mx - mn > 1e-6:
        grad = (grad - mn) / (mx - mn)
    else:
        grad = np.zeros_like(grad)
    return grad.astype(np.float32)

def extract_z_from_sid(sid):
    m = re.search(r'\d+', sid)
    return int(m.group()) if m else 1

def find_image_path(base_img_dir, sid):
    pid = sid[0]
    target = os.path.join(base_img_dir, pid, sid)
    if os.path.exists(target):
        imgs = [
            f for f in os.listdir(target)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
        ]
        if imgs:
            return os.path.join(target, imgs[0])
    return None

if __name__ == "__main__":
    cnt_dir = os.path.join(BASE_DIR, "counts")
    img_dir = os.path.join(BASE_DIR, "images")
    pos_dir = os.path.join(BASE_DIR, "spots")
    lbl_dir = os.path.join(BASE_DIR, "labels")

    files = glob.glob(os.path.join(cnt_dir, "*.tsv.gz"))
    samples = sorted([os.path.basename(f).split('.')[0] for f in files])

    common_genes = None
    for s in tqdm(samples[:5]):
        df = pd.read_csv(
            os.path.join(cnt_dir, f"{s}.tsv.gz"),
            sep='\t',
            index_col=0
        )
        genes = df.index if df.shape[0] > df.shape[1] else df.columns
        genes = set(str(g).upper() for g in genes)
        common_genes = genes if common_genes is None else common_genes & genes

    final_genes = sorted(common_genes)

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        h5f.create_dataset(
            "gene_names",
            data=np.array(final_genes).astype("S")
        )
        h5f.create_dataset(
            "label_names",
            data=np.array(list(LABEL_MAP.keys())).astype("S")
        )
        h5f.create_dataset(
            "label_ids",
            data=np.array(list(LABEL_MAP.values()))
        )

        for sid in tqdm(samples):
            try:
                df_cnt = pd.read_csv(
                    os.path.join(cnt_dir, f"{sid}.tsv.gz"),
                    sep='\t',
                    index_col=0
                )
                if df_cnt.shape[0] > df_cnt.shape[1]:
                    df_cnt = df_cnt.T
                df_cnt.columns = [str(c).upper() for c in df_cnt.columns]
                df_cnt = df_cnt.reindex(columns=final_genes, fill_value=0)

                pos_path = os.path.join(pos_dir, f"{sid}_selection.tsv")
                if not os.path.exists(pos_path):
                    continue
                df_pos = pd.read_csv(pos_path, sep='\t')
                x = np.round(df_pos['x']).astype(int)
                y = np.round(df_pos['y']).astype(int)
                df_pos.index = x.astype(str) + 'x' + y.astype(str)

                lbl_path = os.path.join(lbl_dir, f"{sid}_labeled_coordinates.tsv")
                df_lbl = None
                if os.path.exists(lbl_path):
                    tmp = pd.read_csv(lbl_path, sep='\t')
                    tmp = tmp.dropna(subset=['x', 'y'])
                    if not tmp.empty:
                        lx = np.round(tmp['x']).astype(int)
                        ly = np.round(tmp['y']).astype(int)
                        tmp.index = lx.astype(str) + 'x' + ly.astype(str)
                        df_lbl = tmp

                img_path = find_image_path(img_dir, sid)
                if not img_path:
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                valid = df_cnt.index.intersection(df_pos.index)
                if len(valid) == 0:
                    df_pos.index = y.astype(str) + 'x' + x.astype(str)
                    valid = df_cnt.index.intersection(df_pos.index)
                if len(valid) == 0:
                    continue

                expr = df_cnt.loc[valid].values
                coords = df_pos.loc[valid, ['pixel_x', 'pixel_y']].values

                labels = np.full(len(valid), -1)
                if df_lbl is not None:
                    inter = valid.intersection(df_lbl.index)
                    if len(inter) > 0:
                        s = pd.Series(-1, index=valid)
                        s.loc[inter] = (
                            df_lbl.loc[inter, 'label']
                            .map(LABEL_MAP)
                            .fillna(0)
                            .astype(int)
                        )
                        labels = s.values

                z = (extract_z_from_sid(sid) - 1) / 6.0
                x_norm = coords[:, 0] / w
                y_norm = coords[:, 1] / h
                coords_3d = np.stack(
                    [x_norm, y_norm, np.full_like(x_norm, z)],
                    axis=1
                )

                pad = PATCH_SIZE // 2
                img_pad = cv2.copyMakeBorder(
                    img,
                    pad,
                    pad,
                    pad,
                    pad,
                    cv2.BORDER_REFLECT
                )

                patches = []
                grads = []
                for px, py in np.round(coords).astype(int):
                    px += pad
                    py += pad
                    patch = img_pad[
                        py - pad:py + pad,
                        px - pad:px + pad
                    ]
                    if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                    patches.append(patch)
                    grads.append(compute_sobel_gradient(patch))

                grp = h5f.create_group(sid)
                grp.create_dataset(
                    "patches",
                    data=np.array(patches, np.uint8),
                    compression="lzf"
                )
                grp.create_dataset(
                    "sobel_gradients",
                    data=np.array(grads, np.float32),
                    compression="lzf"
                )
                grp.create_dataset(
                    "coords_3d",
                    data=coords_3d.astype(np.float32)
                )
                grp.create_dataset(
                    "expression",
                    data=expr.astype(np.float32),
                    compression="lzf"
                )
                grp.create_dataset(
                    "labels",
                    data=labels.astype(np.int8)
                )

            except Exception:
                pass
