"""
CLIP feature extraction — use a frozen pretrained CLIP as the backbone,
then train a lightweight classifier on top. Fast to iterate, strong baseline,
and well under the <2B parameter limit (open_clip ViT-B/32 is ~150M params).
"""
import torch
import open_clip
from PIL import Image


class ClipFeatureExtractor:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, img: Image.Image) -> torch.Tensor:
        """Returns a single L2-normalized embedding vector for one PIL image."""
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()

    @torch.no_grad()
    def extract_batch(self, imgs: list[Image.Image]) -> torch.Tensor:
        """Batched extraction for speed."""
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in imgs]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()


if __name__ == "__main__":
    # smoke test
    extractor = ClipFeatureExtractor()
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    emb = extractor.extract(dummy)
    print(f"Embedding shape: {emb.shape}, device used: {extractor.device}")
