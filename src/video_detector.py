"""
Video deepfake detector: samples frames, runs the image pipeline on each,
and adds a temporal-consistency signal (real videos have smoothly-varying
frame-to-frame scores; frame-stitched/face-swapped deepfakes often flicker).
"""

import cv2
import numpy as np
from PIL import Image

from .config import VIDEO_FRAME_SAMPLE_RATE, VIDEO_MAX_FRAMES
from .combined_score import score_image


def analyze_video(video_path: str, image_detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_scores = []
    frame_idx = 0
    analyzed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_FRAME_SAMPLE_RATE == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result = score_image(pil_img, image_detector)
            frame_scores.append(result["final_score"])
            analyzed += 1
            if analyzed >= VIDEO_MAX_FRAMES:
                break
        frame_idx += 1

    cap.release()

    if not frame_scores:
        return {"error": "No frames could be analyzed."}

    frame_scores = np.array(frame_scores)
    mean_score = float(frame_scores.mean())
    # Temporal inconsistency: high frame-to-frame variance is itself suspicious
    temporal_variance = float(np.var(np.diff(frame_scores))) if len(frame_scores) > 1 else 0.0
    temporal_flag = temporal_variance > 0.02  # flicker threshold, tune after testing

    final_score = float(np.clip(mean_score + (0.1 if temporal_flag else 0.0), 0, 1))

    return {
        "frames_analyzed": analyzed,
        "per_frame_scores": frame_scores.tolist(),
        "mean_frame_score": mean_score,
        "temporal_variance": temporal_variance,
        "temporal_flicker_detected": temporal_flag,
        "final_score": final_score,
    }
