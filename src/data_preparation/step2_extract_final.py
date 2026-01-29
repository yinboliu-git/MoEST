import os
import torch
import timm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ================= Configuration Area =================
# Use environment variable or default relative path
H5_PATH = os.environ.get('H5_PATH', './data/openST/suppl/HNSCC_3D_Final_RawCounts.h5')
MODEL_DIR = os.environ.get('MODEL_DIR', './basemodel/UNI')
BATCH_SIZE = 256
# Key modification: Set to 4 for stability; if error occurs set to 0
NUM_WORKERS = 4

# ================= Fixed Dataset =================
class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        # Note: No longer passing opened dataset object, but path and key name
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform

        # Open once to get length, then close
        with h5py.File(h5_path, 'r') as f:
            self.length = f[section_name]['patches'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Key fix: Open file independently within each worker
        # This supports multi-process reading without deadlock
        with h5py.File(self.h5_path, 'r') as f:
            # Read data
            img = f[self.section_name]['patches'][idx]

        # Preprocessing
        if self.transform:
            img = self.transform(img)
        return img

# ================= Model Loading =================
def load_uni_model_local(model_dir, device):
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    print(f"Loading model: {model_dir}")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# ================= Main Program =================
if __name__ == "__main__":
    # 1. Force check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("Critical error: PyTorch cannot detect GPU! Please check environment installation.")

    DEVICE = "cuda"
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")

    uni_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    model = load_uni_model_local(MODEL_DIR, DEVICE)

    print(f"Processing data: {H5_PATH}")

    # Get all section names
    with h5py.File(H5_PATH, 'r') as f:
        sections = [k for k in f.keys() if k.startswith('S')]

    # Process loop
    # Note: We don't open h5 outside the loop, open again each time for writing
    for sec_name in tqdm(sections, desc="Extraction progress"):

        # Check if already exists
        with h5py.File(H5_PATH, 'r') as f:
            if 'uni_features' in f[sec_name]:
                print(f"Skipping {sec_name} (already completed)")
                continue
            n_samples = f[sec_name]['patches'].shape[0]

        if n_samples == 0: continue

        # Use fixed Dataset
        dataset = H5PatchDataset(H5_PATH, sec_name, transform=uni_transform)

        # If H5 filesystem doesn't support concurrency, change num_workers to 0
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        features_list = []
        with torch.inference_mode():
            for batch_imgs in loader:
                batch_imgs = batch_imgs.to(DEVICE) # Ensure data is on GPU
                feats = model(batch_imgs)
                features_list.append(feats.cpu().numpy())

        if len(features_list) > 0:
            all_feats = np.concatenate(features_list, axis=0)

            # Open H5 in write mode
            with h5py.File(H5_PATH, 'r+') as f:
                f[sec_name].create_dataset("uni_features", data=all_feats, compression="lzf")

    print("\nComplete!")
