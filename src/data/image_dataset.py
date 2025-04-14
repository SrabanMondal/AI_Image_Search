from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.transform(image)
            return {"path": image_path, "image": processed_image}
        except Exception as e:
            print(f"Skipping invalid image: {image_path}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "path": [b["path"] for b in batch],
        "image": torch.stack([b["image"] for b in batch])
    }