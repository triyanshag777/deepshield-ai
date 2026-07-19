"""
Additional forensic checks that matter for investigative use.
"""

import hashlib
from PIL import Image, ExifTags
import numpy as np


def sha256_of_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _convert_gps_to_decimal(gps_coords, ref):
    try:
        degrees, minutes, seconds = gps_coords
        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


def _extract_gps(pil_image: Image.Image):
    try:
        exif = pil_image.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps_ifd:
            return None
        gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat = gps_tags.get("GPSLatitude")
        lat_ref = gps_tags.get("GPSLatitudeRef", "N")
        lon = gps_tags.get("GPSLongitude")
        lon_ref = gps_tags.get("GPSLongitudeRef", "E")
        if lat is None or lon is None:
            return None
        latitude = _convert_gps_to_decimal(lat, lat_ref)
        longitude = _convert_gps_to_decimal(lon, lon_ref)
        if latitude is None or longitude is None:
            return None
        return {"latitude": latitude, "longitude": longitude}
    except Exception:
        return None


def exif_report(pil_image: Image.Image):
    exif_raw = pil_image.getexif()
    if not exif_raw or len(exif_raw) == 0:
        return {
            "has_exif": False,
            "fields": {},
            "gps": None,
            "suspicion_note": "No EXIF metadata found. Common in AI-generated images "
                               "(never passed through a camera ISP pipeline), screenshots, "
                               "or images stripped of metadata during editing/sharing.",
        }

    fields = {}
    for tag_id, value in exif_raw.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        fields[str(tag)] = str(value)

    gps = _extract_gps(pil_image)

    software = fields.get("Software", "")
    suspicious_software = any(
        kw in software.lower() for kw in [
            "photoshop", "gimp", "midjourney", "stable diffusion", "dall-e", "dalle",
            "google photos", "magic eraser", "magic editor", "pixel", "snapseed",
            "lightroom", "canva", "picsart", "facetune", "remini", "ai", "generative",
        ]
    )

    note = "EXIF present."
    if suspicious_software:
        note += f" Software tag ('{software}') indicates image editing/generation tool involvement."
    if "Make" not in fields and "Model" not in fields:
        note += " No camera make/model recorded — inconsistent with a direct camera capture."
    if gps:
        note += f" GPS location found: {gps['latitude']}, {gps['longitude']}."
    if not suspicious_software:
        note += (" Note: presence of EXIF is supporting evidence toward authenticity, not proof — "
                  "some AI/in-app editors (e.g. Google Photos Magic Editor) preserve the original "
                  "camera EXIF even after AI-based edits, since they modify pixels in place.")

    key_fields = {}
    for label, tag in [("Camera Make", "Make"), ("Camera Model", "Model"),
                        ("Date Taken", "DateTime"), ("Software", "Software")]:
        if tag in fields:
            key_fields[label] = fields[tag]

    return {
        "has_exif": True,
        "fields": fields,
        "key_fields": key_fields,
        "gps": gps,
        "suspicion_note": note,
    }


def perceptual_hash(pil_image: Image.Image, hash_size: int = 16) -> str:
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
