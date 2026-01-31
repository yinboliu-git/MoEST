import os
import torch
import timm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

H5_PATH = "/path/to/data/DATASET.h5"
MODEL_DIR = "/path/to/model/UNI"

BATCH_SIZE = 256
NUM_WORKERS = 0

warnings.filterwarnings("ignore", category=UserWarning)

class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.h5_path = h5_path
        self.section_name = section_name
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.length = f[section_name]['patches'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f[self.section_name]['patches'][idx]
        if self.transform:
            img = self.transform(img)
        return img

def load_uni_model(model_dir):
    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    device = torch.device("cuda")

    model = load_uni_model(MODEL_DIR)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    with h5py.File(H5_PATH, 'r') as f:
        keys = list(f.keys())
        sections = [
            k for k in keys
            if k[0].isalpha() and 'gene' not in k and 'label' not in k
        ]
        sections.sort()

    success_count = 0

    for sec in tqdm(sections):
        with h5py.File(H5_PATH, 'r') as f:
            if 'uni_features' in f[sec]:
                continue
            n = f[sec]['patches'].shape[0]

        if n == 0:
            continue

        dataset = H5PatchDataset(H5_PATH, sec, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        feats_all = []

        with torch.inference_mode():
            for imgs in loader:
                imgs = imgs.to(device, non_blocking=True)
                feats = model(imgs)
                feats_all.append(feats.cpu().numpy())

        if len(feats_all) > 0:
            feats_all = np.concatenate(feats_all, axis=0)
            with h5py.File(H5_PATH, 'r+') as f:
                f[sec].create_dataset(
                    "uni_features",
                    data=feats_all,
                    compression="lzf"
                )
            success_count += 1
