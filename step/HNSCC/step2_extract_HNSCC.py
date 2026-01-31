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

H5_PATH = "/path/to/data/DATASET_RAWCOUNTS.h5"
MODEL_DIR = "/path/to/model/UNI"

BATCH_SIZE = 512
NUM_WORKERS = 0

class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform
        cv2.setNumThreads(0)
        with h5py.File(h5_path, "r") as f:
            self.length = f[section_name]["patches"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f[self.section_name]["patches"][idx]
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        return img

def load_uni_model_local(model_dir, device):
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    return model

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    DEVICE = "cuda:0"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    model = load_uni_model_local(MODEL_DIR, DEVICE)

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError("H5 file not found")

    with h5py.File(H5_PATH, "r") as f:
        sections = [k for k in f.keys() if k.startswith("S")]
        sections.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

    for sec_name in tqdm(sections):
        with h5py.File(H5_PATH, "r") as f:
            if "uni_features" in f[sec_name]:
                continue
            n_samples = f[sec_name]["patches"].shape[0]

        if n_samples == 0:
            continue

        dataset = H5PatchDataset(H5_PATH, sec_name, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        features = []

        with torch.inference_mode():
            for imgs in loader:
                imgs = imgs.to(DEVICE)
                feats = model(imgs)
                features.append(feats.cpu().numpy())

        if features:
            all_feats = np.concatenate(features, axis=0)
            with h5py.File(H5_PATH, "r+") as f:
                if "uni_features" in f[sec_name]:
                    del f[sec_name]["uni_features"]
                f[sec_name].create_dataset(
                    "uni_features",
                    data=all_feats,
                    compression="lzf",
                )
