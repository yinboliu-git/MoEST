import os
import torch
import torch.nn as nn
import timm
import h5py
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ================= Configuration Area =================
# Use environment variable or default relative path
H5_PATH = os.environ.get('H5_PATH', './data/openST/suppl/HNSCC_3D_Final_RawCounts_new.h5')
MODEL_DIR = os.environ.get('MODEL_DIR', './basemodel/UNI')

# Optimized for multi-GPU (GPU 5+6): Increase Batch Size for better utilization
BATCH_SIZE = 512
NUM_WORKERS = 0   # Increase worker count to keep up with multi-GPU computation speed

# ================= Dataset Definition =================
class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform

        # Prevent cv2 multi-threading conflicts with DataLoader causing freeze
        cv2.setNumThreads(0)

        # Open once to get length
        with h5py.File(h5_path, 'r') as f:
            self.length = f[section_name]['patches'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file independently for concurrent reading
        with h5py.File(self.h5_path, 'r') as f:
            img = f[self.section_name]['patches'][idx]

        # Double insurance: Prevent size inconsistency causing crash
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))

        # Preprocessing
        if self.transform:
            img = self.transform(img)
        return img

# ================= Model Loading (with multi-GPU logic) =================
def load_uni_model_local(model_dir, device):
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    print(f"Loading UNI model: {model_dir}")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    # 1. Move to main device first
    model.to(device)

    # 2. Key modification: Activate multi-GPU parallelism (DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"[Multi-GPU mode activated] Detected {torch.cuda.device_count()} GPUs, enabling parallel acceleration...")
        model = nn.DataParallel(model)
    else:
        print(f"[Single-GPU mode] Only detected 1 GPU")

    model.eval()
    return model

# ================= Main Program =================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Critical error: PyTorch cannot detect GPU!")

    # Always specify cuda:0, as CUDA_VISIBLE_DEVICES will remap IDs
    DEVICE = "cuda:0"

    # Print current visible GPU list
    print(f"Available GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    uni_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    model = load_uni_model_local(MODEL_DIR, DEVICE)

    print(f"Preparing to process file: {H5_PATH}")

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"File not found: {H5_PATH}")

    with h5py.File(H5_PATH, 'r') as f:
        # Filter out non-section keys (like gene_names), only keep S-prefixed ones
        sections = [k for k in f.keys() if k.startswith('S')]
        # Sort by S1, S2... numerically
        sections.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    print(f"Sections to process: {sections}")

    for sec_name in tqdm(sections, desc="Extraction progress"):

        # 1. Check if already processed
        with h5py.File(H5_PATH, 'r') as f:
            if 'uni_features' in f[sec_name]:
                continue
            n_samples = f[sec_name]['patches'].shape[0]

        if n_samples == 0: continue

        # 2. Prepare DataLoader
        dataset = H5PatchDataset(H5_PATH, sec_name, transform=uni_transform)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        features_list = []

        # 3. Inference (automatic multi-GPU)
        with torch.inference_mode():
            for batch_imgs in loader:
                batch_imgs = batch_imgs.to(DEVICE)
                # DataParallel will automatically split batch to GPU 5 and 6
                feats = model(batch_imgs)
                features_list.append(feats.cpu().numpy())

        # 4. Write to H5
        if len(features_list) > 0:
            all_feats = np.concatenate(features_list, axis=0)

            with h5py.File(H5_PATH, 'r+') as f:
                # Use lzf compression to store features
                if 'uni_features' in f[sec_name]:
                    del f[sec_name]['uni_features'] # Delete first if bad data exists
                f[sec_name].create_dataset("uni_features", data=all_feats, compression="lzf")

    print("\nOpenST feature extraction fully complete!")
