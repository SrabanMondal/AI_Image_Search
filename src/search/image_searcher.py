import torch
import torch.nn.functional as F
from src.data.image_dataset import ImageDataset, collate_fn
from src.models.clip_model import CLIPModel
from torch.utils.data import DataLoader

class ImageSearcher:
    def __init__(self, clip_model, image_paths, batch_size=4):
        self.clip_model = clip_model
        self.dataset = ImageDataset(image_paths, transform=clip_model.preprocess)
        self.data_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    def search(self, query_text, top_k=10):
        text_embedding = self.clip_model.encode_text(query_text)
        ranked_scores = []
        for batch in self.data_loader:
            if batch is None:
                continue
            paths = batch["path"]
            images = batch["image"].to(self.clip_model.device)
            image_embeddings = self.clip_model.encode_image_batch(images)
            scores = F.cosine_similarity(text_embedding, image_embeddings).tolist()
            ranked_scores.extend(zip(paths, scores))
        return sorted(ranked_scores, key=lambda x: x[1], reverse=True)[:top_k]