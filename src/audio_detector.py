"""
Audio deepfake / synthetic voice detector using a pretrained Wav2Vec2 classifier.
"""

import torch
import torch.nn.functional as F
import torchaudio

from .config import AUDIO_MODEL, DEVICE


class AudioDetector:
    def __init__(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        print(f"[AudioDetector] Loading {AUDIO_MODEL} ...")
        self.extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        self.model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL)
        self.model.to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.target_sr = self.extractor.sampling_rate

    @torch.no_grad()
    def predict(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        inputs = self.extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

        fake_prob = 0.0
        found = False
        for idx, label in self.id2label.items():
            if "fake" in str(label).lower():
                fake_prob += probs[int(idx)].item()
                found = True
        if not found:
            fake_prob = probs[1].item() if probs.shape[0] > 1 else probs[0].item()

        return {"fake_prob": fake_prob, "raw_probs": {self.id2label[i]: probs[i].item() for i in range(len(probs))}}
