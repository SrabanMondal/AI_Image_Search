import torch
import clip
from PIL import Image

class CLIPModel:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def encode_text(self, text):
        tokenized = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokenized)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def encode_image(self, image):
        processed = self.preprocess(Image.open(image).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(processed)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def encode_image_batch(self, images):
        with torch.no_grad():
            embeddings = self.model.encode_image(images)
        return embeddings / embeddings.norm(dim=-1, keepdim=True)