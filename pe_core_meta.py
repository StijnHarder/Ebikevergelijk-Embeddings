import torch
from PIL import Image
import perception_models.core.vision_encoder.pe as pe
import perception_models.core.vision_encoder.transforms as transforms
import requests
import io

class PeCoreModelLoader:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device)
        self.processor = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
    
    def get_image_tensor(self, url: str):
        try:
            r = requests.get(url, timeout=10)
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
            return self.processor(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[ERROR] Failed loading image {url}: {e}")
            return None
    
    def generate_joint_embedding(self, title: str, image_url: str):
        image_tensor = self.get_image_tensor(image_url)
        if image_tensor is None:
            return None
        
        text_tokens = self.tokenizer(title).to(self.device)

        with torch.no_grad():
            img_emb, txt_emb, _ = self.model(image_tensor, text_tokens)

        # Shapes: [1, D]
        combined = torch.cat([img_emb, txt_emb], dim=1)

        # Convert for Supabase
        return combined.squeeze(0).cpu().float().tolist()
