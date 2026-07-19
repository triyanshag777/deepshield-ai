"""
Image deepfake detector: ensembles multiple pretrained HuggingFace models.
Uses ViT and SigLIP architectures fine-tuned for real-vs-fake classification.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .config import IMAGE_MODELS, DEVICE


class ImageEnsembleDetector:
    def __init__(self):
        self.models = []
        for repo_id, arch_hint, weight in IMAGE_MODELS:
            print(f"[ImageEnsembleDetector] Loading {repo_id} ...")
            processor = AutoImageProcessor.from_pretrained(repo_id)
            model = AutoModelForImageClassification.from_pretrained(repo_id)
            model.to(DEVICE)
            model.eval()
            self.models.append({
                "repo_id": repo_id,
                "processor": processor,
                "model": model,
                "weight": weight,
                "id2label": model.config.id2label,
            })

    def _fake_prob_from_logits(self, logits, id2label):
        """Map arbitrary label ordering (fake/real, 0_real/1_fake, etc.) to a single P(fake)."""
        probs = F.softmax(logits, dim=-1).squeeze(0)
        fake_prob = 0.0
        found = False
        for idx, label in id2label.items():
            label_l = str(label).lower()
            if "fake" in label_l or "1_fake" in label_l or label_l == "1":
                fake_prob += probs[int(idx)].item()
                found = True
        if not found:
            # fallback: assume index 1 is "fake"
            fake_prob = probs[1].item() if probs.shape[0] > 1 else probs[0].item()
        return fake_prob

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """Returns dict: {per_model: {...}, ensemble_fake_prob: float}"""
        image = image.convert("RGB")
        results = {"per_model": {}}
        weighted_sum = 0.0
        weight_total = 0.0

        for m in self.models:
            inputs = m["processor"](images=image, return_tensors="pt").to(DEVICE)
            outputs = m["model"](**inputs)
            fake_prob = self._fake_prob_from_logits(outputs.logits, m["id2label"])
            results["per_model"][m["repo_id"]] = fake_prob
            weighted_sum += fake_prob * m["weight"]
            weight_total += m["weight"]

        results["ensemble_fake_prob"] = weighted_sum / weight_total if weight_total else 0.0
        return results
