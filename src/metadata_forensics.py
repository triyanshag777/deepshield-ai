"""
Additional forensic checks that matter for investigative use:
1. EXIF metadata analysis — missing/stripped EXIF is common in AI-generated
   or screenshotted images; inconsistent EXIF (e.g. software tag showing an
   editor, mismatched timestamps) is a manipulation signal.
2. Perceptual hash — flags whether the "new evidence" image is actually a
   duplicate or near-duplicate of something already seen (recycled photo
   passed off as a new incident), which matters a lot for case triage.
"""

import hashlib
from PIL import Image, ExifTags
import numpy as np


def sha256_of_file(file_path: str) -> str:
    """Chain-of-custody hash: proves the file wasn't altered after analysis."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_report(pil_image: Image.Image):
    """
    Returns a dict summarizing EXIF findings + a suspicion flag.
    No EXIF at all is common for AI-generated images (they never touched a
    camera sensor / ISP pipeline) but also common for legitimately
    re-saved/screenshotted real photos — so this is a supporting signal,
    not standalone proof.
    """
    exif_raw = pil_image.getexif()
    if not exif_raw or len(exif_raw) == 0:
        return {
            "has_exif": False,
            "fields": {},
            "suspicion_note": "No EXIF metadata found. Common in AI-generated images "
                               "(never passed through a camera ISP pipeline), screenshots, "
                               "or images stripped of metadata during editing/sharing.",
        }

    fields = {}
    for tag_id, value in exif_raw.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        fields[str(tag)] = str(value)

    software = fields.get("Software", "")
    suspicious_software = any(
        kw in software.lower() for kw in ["photoshop", "gimp", "midjourney", "stable diffusion", "dall-e", "ai"]
    )

    note = "EXIF present."
    if suspicious_software:
        note += f" Software tag ('{software}') indicates image editing/generation tool involvement."
    if "Make" not in fields and "Model" not in fields:
        note += " No camera make/model recorded — inconsistent with a direct camera capture."

    return {"has_exif": True, "fields": fields, "suspicion_note": note}


def perceptual_hash(pil_image: Image.Image, hash_size: int = 16) -> str:
    """
    Simple average-hash perceptual hash (no extra dependency beyond Pillow/numpy).
    Two images with a small Hamming distance between hashes are near-duplicates.
    """
    img = pil_image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return format(hash_int, f"0{hash_size*hash_size//4}x")


def hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    int_a, int_b = int(hash_a, 16), int(hash_b, 16)
    return bin(int_a ^ int_b).count("1")
