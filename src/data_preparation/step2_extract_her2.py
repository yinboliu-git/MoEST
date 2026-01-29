import os
import torch
import timm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

# ================= Configuration Area =================
# Use environment variable or default relative path
H5_PATH = os.environ.get('H5_PATH', './data/her2st/HER2_3D_Final_RawCounts.h5')
MODEL_DIR = os.environ.get('MODEL_DIR', './basemodel/UNI')

# Performance parameters
BATCH_SIZE = 256
# Key setting: Set to 0 to prevent H5 deadlock. A100 inference is fast enough with single process.
NUM_WORKERS = 0

warnings.filterwarnings("ignore", category=UserWarning)

# ================= Dataset Definition =================
class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform
        # Only read length, don't keep handle
        with h5py.File(h5_path, 'r') as f:
            self.length = f[section_name]['patches'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file each time for multi-process/thread safety (though using single process here)
        with h5py.File(self.h5_path, 'r') as f:
            img = f[self.section_name]['patches'][idx] # Read uint8 image

        if self.transform:
            img = self.transform(img)
        return img

# ================= Model Loading =================
def load_uni_model(model_dir):
    print(f"Loading UNI model: {model_dir}")
    weight_path = os.path.join(model_dir, "pytorch_model.bin")

    # Create ViT-Large model
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )

    # Load weights
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model

# ================= Main Program =================
if __name__ == "__main__":
    # 1. GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected, please check environment!")

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # 2. Load model
    model = load_uni_model(MODEL_DIR)
    model.to(device)
    model.eval()

    # 3. Preprocessing (ImageNet normalization)
    uni_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print(f"Starting data processing: {H5_PATH}")

    # 4. Get list of samples to process
    with h5py.File(H5_PATH, 'r') as f:
        # Filter out gene_names, labels and other non-sample keys
        all_keys = list(f.keys())
        sections = [k for k in all_keys if k[0].isalpha() and 'gene' not in k and 'label' not in k]
        sections.sort()

    print(f"Found {len(sections)} samples to check...")

    # 5. Process loop
    success_count = 0
    for sec_name in tqdm(sections, desc="Extraction progress"):

        # Check if features already exist
        with h5py.File(H5_PATH, 'r') as f:
            if 'uni_features' in f[sec_name]:
                # Skip if already exists (saves time)
                continue

            # Get patch count
            n_samples = f[sec_name]['patches'].shape[0]

        if n_samples == 0: continue

        # Initialize DataLoader
        dataset = H5PatchDataset(H5_PATH, sec_name, transform=uni_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        features_list = []

        # Inference
        with torch.inference_mode():
            for batch_imgs in loader:
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                feats = model(batch_imgs) # (B, 1024)
                features_list.append(feats.cpu().numpy())

        # Save
        if len(features_list) > 0:
            all_feats = np.concatenate(features_list, axis=0)

            with h5py.File(H5_PATH, 'r+') as f:
                # Write features
                f[sec_name].create_dataset("uni_features", data=all_feats, compression="lzf")
                success_count += 1

    print(f"\nExtraction complete! Processed {success_count} new samples.")
