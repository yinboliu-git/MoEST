import os
import torch
import timm
import h5py
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

H5_PATH = "/path/to/data/MISAR_XYT.h5"
MODEL_DIR = "/path/to/model/UNI"

BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class H5PatchDataset(Dataset):
    def __init__(self, h5_path, section_name, transform=None):
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            self.patches = f[section_name]["patches"][:]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        if self.transform:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError("Input file not found")

    try:
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu"),
            strict=True,
        )
        model.to(DEVICE)
        model.eval()
    except Exception:
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
        model.to(DEVICE)
        model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    with h5py.File(H5_PATH, "r+") as f:
        samples = [k for k in f.keys() if k.startswith("E")]
        samples.sort()

        for sec in tqdm(samples):
            if "uni_features" in f[sec]:
                continue

            dataset = H5PatchDataset(H5_PATH, sec, transform=transform)
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            features = []
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(DEVICE)
                    feats = model(imgs)
                    features.append(feats.cpu().numpy())

            if features:
                all_feats = np.concatenate(features, axis=0)
                f[sec].create_dataset(
                    "uni_features",
                    data=all_feats.astype(np.float32),
                )
