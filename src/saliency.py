"""
Explainability: occlusion sensitivity map.
Slides a gray patch across the image, re-scores each occluded version, and
records how much the fake-probability *drops* when each region is hidden.
Regions that cause the biggest drop are the regions the model is relying on
to call the image fake - visualized as a heatmap overlay.

Architecture-agnostic (works with any black-box classifier), which matters
because our ensemble mixes ViT and SigLIP backbones with different internals.
"""

import numpy as np
import cv2
from PIL import Image


def occlusion_heatmap(pil_image: Image.Image, image_detector, patch_size: int = 32, stride: int = 24):
    image = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(image)
    h, w, _ = arr.shape

    baseline = image_detector.predict(image)["ensemble_fake_prob"]
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = arr.copy()
            occluded[y:y + patch_size, x:x + patch_size] = 127  # gray patch
            occ_img = Image.fromarray(occluded)
            occ_score = image_detector.predict(occ_img)["ensemble_fake_prob"]

            drop = max(0.0, baseline - occ_score)  # how much confidence dropped
            heat[y:y + patch_size, x:x + patch_size] += drop
            counts[y:y + patch_size, x:x + patch_size] += 1

    counts[counts == 0] = 1
    heat = heat / counts
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    return overlay, baseline
