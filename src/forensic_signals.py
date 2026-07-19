"""
Classical + signal-processing forensic checks using OpenCV / NumPy.
These are model-agnostic signals that complement the deep learning classifiers -
AI-generated images often show up here even when they fool a single classifier.
"""

import cv2
import numpy as np
from PIL import Image
import io


def error_level_analysis(pil_image: Image.Image, quality: int = 90):
    """
    ELA: re-compress the image at a known JPEG quality and diff against the original.
    Real camera photos have fairly uniform compression noise; GAN/diffusion images
    often show unnaturally smooth regions or inconsistent block artifacts.
    Returns (anomaly_score 0-1, ela_visual as np.uint8 BGR image for display).
    """
    rgb = pil_image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(rgb).astype(np.int16)
    recompressed_arr = np.array(recompressed).astype(np.int16)

    diff = np.abs(orig_arr - recompressed_arr).astype(np.uint8)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Normalize score: high variance of error residual + low mean can indicate
    # over-smooth AI generation; very high mean can indicate heavy manipulation.
    mean_err = gray_diff.mean()
    std_err = gray_diff.std()
    # Heuristic scaling into 0-1 anomaly score (tuned empirically, adjust after testing)
    anomaly = np.clip((std_err / 25.0) * 0.5 + (mean_err / 40.0) * 0.5, 0, 1)

    ela_visual = cv2.applyColorMap(
        cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    return float(anomaly), ela_visual


def fft_noise_residual(pil_image: Image.Image):
    """
    Frequency-domain analysis: GAN/diffusion upsampling leaves characteristic
    periodic artifacts in the high-frequency spectrum that real camera sensor
    noise doesn't have.
    Returns (anomaly_score 0-1, spectrum_visual as np.uint8 image).
    """
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Look at high-frequency ring energy vs low-frequency center energy.
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low_mask = dist < (min(h, w) * 0.10)
    high_mask = (dist >= (min(h, w) * 0.35)) & (dist < (min(h, w) * 0.5))

    low_energy = magnitude[low_mask].mean()
    high_energy = magnitude[high_mask].mean()

    # Real photos: high-freq energy decays smoothly from center.
    # AI-generated: unusually flat or spiky high-freq energy relative to low-freq.
    ratio = high_energy / (low_energy + 1e-6)
    anomaly = float(np.clip((ratio - 0.35) / 0.4, 0, 1))

    spectrum_visual = cv2.applyColorMap(
        cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    return anomaly, spectrum_visual


_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def face_symmetry_score(pil_image: Image.Image):
    """
    Detects the largest face, flips it, and measures left/right asymmetry.
    GAN faces frequently show subtle unnatural symmetry OR asymmetric artifacts
    (mismatched earrings, warped glasses, blended hairlines).
    Returns (anomaly_score 0-1, face_found: bool).
    """
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        return 0.0, False

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (200, 200))

    flipped = cv2.flip(face, 1)
    diff = cv2.absdiff(face, flipped).astype(np.float32)
    asym = diff.mean() / 255.0

    # Very low asymmetry (near-perfect mirror) OR very high asymmetry both raise suspicion.
    anomaly = float(np.clip(abs(asym - 0.12) / 0.12, 0, 1))
    return anomaly, True
