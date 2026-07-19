"""
DeepShield AI - central config
All model IDs are real, public HuggingFace repos (verified July 2026).
Swap any of these freely if you find a better one on the Hub before your demo.
"""

import torch

# ---- Device selection (Apple Silicon M-series uses "mps") ----
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ---- Pretrained models (HuggingFace Hub IDs) ----
IMAGE_MODELS = [
    # (repo_id, architecture_hint, weight_in_ensemble)
    ("prithivMLmods/Deep-Fake-Detector-v2-Model", "vit", 0.5),
    ("prithivMLmods/deepfake-detector-model-v1", "siglip", 0.5),
]

AUDIO_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"

# ---- Ensemble weights across modalities/signals for the final image score ----
SIGNAL_WEIGHTS = {
    "model_ensemble": 0.55,   # pretrained classifier vote
    "ela": 0.20,               # error level analysis (compression inconsistency)
    "fft_noise": 0.15,         # frequency-domain residual noise pattern
    "face_symmetry": 0.10,     # left/right facial symmetry anomaly
}

# ---- Video sampling ----
VIDEO_FRAME_SAMPLE_RATE = 15   # analyze 1 frame every N frames
VIDEO_MAX_FRAMES = 40          # cap frames analyzed per video (speed)

# ---- Classification thresholds ----
FAKE_THRESHOLD = 0.5
