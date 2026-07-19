"""
Combines the pretrained-model ensemble score with the classical forensic
signals (ELA, FFT noise, face symmetry) into one final weighted verdict.
This is the core "much more accuracy" trick: no single signal has to be
perfect, because they rarely all fail on the same image at once.
"""

from PIL import Image
from .config import SIGNAL_WEIGHTS
from .forensic_signals import error_level_analysis, fft_noise_residual, face_symmetry_score


def score_image(pil_image: Image.Image, image_detector, return_visuals: bool = False):
    model_result = image_detector.predict(pil_image)
    model_score = model_result["ensemble_fake_prob"]

    ela_score, ela_visual = error_level_analysis(pil_image)
    fft_score, fft_visual = fft_noise_residual(pil_image)
    sym_score, face_found = face_symmetry_score(pil_image)

    weights = dict(SIGNAL_WEIGHTS)
    if not face_found:
        # redistribute the face_symmetry weight to the model ensemble if no face detected
        weights["model_ensemble"] += weights.pop("face_symmetry")
        sym_score = 0.0

    final_score = (
        weights.get("model_ensemble", 0) * model_score
        + weights.get("ela", 0) * ela_score
        + weights.get("fft_noise", 0) * fft_score
        + weights.get("face_symmetry", 0) * sym_score
    )

    result = {
        "final_score": float(final_score),
        "model_ensemble_score": model_score,
        "per_model_scores": model_result["per_model"],
        "ela_score": ela_score,
        "fft_noise_score": fft_score,
        "face_symmetry_score": sym_score,
        "face_found": face_found,
    }
    if return_visuals:
        result["ela_visual"] = ela_visual
        result["fft_visual"] = fft_visual
    return result
