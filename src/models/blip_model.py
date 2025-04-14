from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
class BLIPModel:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            caption = self.model.generate(**inputs, max_length=80)
        return self.processor.decode(caption[0], skip_special_tokens=True)