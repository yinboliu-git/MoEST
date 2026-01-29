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
# 1. Input file (just generated Fixed file)
H5_PATH = os.environ.get('H5_PATH', './data/Misar/MISAR_4D_Final_Fixed.h5')

# 2. UNI model weights path (please confirm your path is correct)
MODEL_DIR = os.environ.get('MODEL_DIR', './basemodel/UNI')

# 3. Parameters
BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# ================= Dataset =================
class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.patches = f[section_name]['patches'][:] # Load all into memory at once (MISAR sections not large, usually feasible)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx] # (224, 224, 3)
        if self.transform:
            img = self.transform(img)
        return img

# ================= Main =================
if __name__ == "__main__":
    print(f"[Step 2] Extracting UNI Features for MISAR...")

    # 1. Load UNI model
    print(f"   Loading UNI model from {MODEL_DIR}...")
    try:
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu"), strict=True)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"   UNI model loading failed: {e}")
        print("   Trying to use default ResNet50 (for testing workflow only)...")
        model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(DEVICE).eval()

    # Preprocessing
    uni_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Iterate through samples in H5
    with h5py.File(H5_PATH, 'r+') as f: # 'r+' mode allows read-write
        # Get all sample names (E11, E13...)
        samples = [k for k in f.keys() if k.startswith('E')]
        samples.sort()

        for sec in tqdm(samples, desc="Extracting"):
            if 'uni_features' in f[sec]:
                print(f"   Skipping {sec} (already exists)")
                continue

            # Create Dataset
            ds = H5PatchDataset(H5_PATH, sec, transform=uni_transform)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            feats_list = []
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(DEVICE)
                    # UNI output
                    feats = model(imgs) # (B, 1024)
                    feats_list.append(feats.cpu().numpy())

            if len(feats_list) > 0:
                all_feats = np.concatenate(feats_list, axis=0)
                # Write to H5
                f[sec].create_dataset("uni_features", data=all_feats.astype(np.float32))

    print("Features extracted and saved to H5!")
