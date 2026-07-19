# 🛡️ DeepShield AI — Forensic Edition

Multimodal detector for AI-generated / manipulated images, video, and audio,
built as a **cyber-cell investigative triage tool** — not just a classifier
demo. It combines pretrained deep learning classifiers (PyTorch + HuggingFace)
with classical forensic signal analysis (OpenCV), a RAG-grounded technical
explanation, chain-of-custody reporting, and a second RAG layer that maps
findings to the actual applicable Indian law (BNS 2023 / IT Act).

## What makes this "next level" vs. a typical hackathon detector
Most student deepfake projects stop at "here's a fake/real label." This one
answers the three questions an actual investigator asks next:
1. **Why do you say it's fake?** RAG-grounded forensic explanation citing
   which signals fired (ELA, FFT, symmetry, classifier ensemble)
2. **Can I trust this file wasn't tampered with after analysis?** SHA-256
   chain-of-custody hash embedded in a generated PDF report
3. **What can I actually do with this?** Legal-RAG layer citing the
   specific BNS/IT Act sections that could apply, plus the cybercrime.gov.in
   reporting portal — this is the part almost no other team will have,
   because it requires knowing the law changed in July 2024 (IPC to BNS),
   which most tutorials online still don't reflect.

**Important framing for your presentation**: pitch this as a *triage and
first-pass investigation aid* that helps a cyber cell prioritize cases and
draft an initial reference note, not as certified forensic evidence or a
replacement for expert analysis. This is the honest, credible framing, and
it's also the one that survives a tough judge's questions.

## Why this is more accurate than a single classifier
Most deepfake detectors are one CNN/ViT trained on one dataset — they miss
generators they weren't trained on. DeepShield instead scores every image on
**four independent signals** and blends them:
1. An **ensemble of 2 pretrained HF classifiers** (different architectures: ViT + SigLIP)
2. **Error Level Analysis** (compression-inconsistency detection)
3. **FFT frequency-domain analysis** (GAN/diffusion upsampling artifacts)
4. **Facial symmetry analysis** (GAN face-generation artifacts)

For video, it adds a **temporal consistency check** across sampled frames
(deepfakes flicker; real video doesn't). For audio, it uses a pretrained
Wav2Vec2 synthetic-voice classifier.

## Setup (MacBook Air M5)

```bash
cd deepshield-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon note: PyTorch will automatically use the `mps` backend
(GPU acceleration) — this is handled in `src/config.py`, no extra setup needed.

First run will download ~1-2GB of pretrained model weights from HuggingFace
(needs internet). After that, everything runs offline.

## Run the demo

```bash
python app.py
```

This opens a Gradio web UI at `http://127.0.0.1:7860` with three tabs:
Image, Video, Audio. Upload a file, hit Analyze — you'll get:
- A verdict + confidence %
- A saliency heatmap (image tab) showing *where* the model focused
- A forensic explanation citing which signals fired and why they matter

## Tonight's build order (fastest path to a working demo)

1. **Now**: `pip install -r requirements.txt`, then run `python app.py` once
   just to trigger the model downloads (do this ASAP, downloads take a while
   on hostel wifi — start it and keep working while it downloads).
2. **Test on 5-10 sample images**: mix of real photos (take some on your
   phone) and AI images (generate a few with any free image generator, or
   pull samples from `thispersondoesnotexist.com` style sites). Sanity-check
   the verdicts make sense.
3. **Tune `FAKE_THRESHOLD` in `src/config.py`** if everything is scoring too
   high or too low — this is normal, pretrained models need threshold
   calibration to your specific test set.
4. **Test video tab** on a short (5-10 sec) clip — this is slower (analyzes
   multiple frames), so test it once and don't over-iterate; lower
   `VIDEO_MAX_FRAMES` in config if it's too slow for your machine.
5. **Test audio tab** if you have time — it's the smallest, most isolated
   module, so it's safe to leave for last or cut if you're short on time.
6. **Prepare 3 demo files in advance**: one obviously-real photo, one
   obviously-AI photo, one video. Don't rely on live uploads during judging —
   have your demo files ready on your desktop.

## What to say to judges (your differentiators)

- "We don't rely on a single model — we ensemble two different architectures
  **and** three classical forensic signals, so a generator that fools one
  detector still gets caught by frequency analysis or compression analysis."
- "Every verdict is explainable — we use retrieval-augmented generation over
  a forensic-indicators knowledge base, so the explanation is grounded in
  real detection literature, not a hallucinated LLM guess."
- "The saliency heatmap shows exactly which region drove the decision —
  this matters for real-world trust and auditability."
- Be honest about limits if asked: pretrained models drift as new generators
  (Sora, newer diffusion models) emerge — mention this as "future work:
  continuous retraining pipeline," it shows maturity, not weakness.

## Project structure

```
deepshield-ai/
├── app.py                          # Gradio demo (run this)
├── requirements.txt
├── src/
│   ├── config.py                   # model IDs, device, weights, thresholds
│   ├── image_detector.py           # pretrained ViT + SigLIP ensemble
│   ├── forensic_signals.py         # OpenCV: ELA, FFT, face symmetry
│   ├── combined_score.py           # weighted fusion of all signals
│   ├── video_detector.py           # frame sampling + temporal check
│   ├── audio_detector.py           # pretrained Wav2Vec2 voice detector
│   ├── saliency.py                 # occlusion-based explainability heatmap
│   ├── rag_explainer.py            # RAG grounded forensic explanation
│   ├── legal_rag.py                # RAG: maps findings to BNS/IT Act sections
│   ├── metadata_forensics.py       # EXIF check, SHA-256 hash, perceptual hash
│   ├── forensic_report.py          # generates the chain-of-custody PDF
│   └── knowledge_base/
│       ├── forensic_indicators.md  # technical RAG knowledge base (editable!)
│       └── legal_sections.md       # legal RAG knowledge base (editable!)
```

## Using the forensic report (Image tab)
1. Upload an image, optionally fill in a Case ID, click **Analyze Image**.
2. Click **Generate Chain-of-Custody PDF Report** — downloads a PDF with the
   verdict, signal breakdown, SHA-256 hash, EXIF notes, and the applicable
   legal reference section. This is your strongest demo artifact — show it
   to judges as the deliverable a cyber cell would actually receive.

## If something breaks at 2am

- **Model download fails / times out**: swap to a smaller model in
  `src/config.py` — e.g. drop to a single model in `IMAGE_MODELS` instead of
  two, ensemble still works with one entry.
- **`mps` errors on Mac**: force CPU by editing `get_device()` in
  `src/config.py` to `return torch.device("cpu")` — slower but always works.
- **Gradio port busy**: `demo.launch(server_port=7861)` in `app.py`.
- **Video analysis too slow**: lower `VIDEO_MAX_FRAMES` and
  `VIDEO_FRAME_SAMPLE_RATE` in `src/config.py` (fewer frames = faster).
- **Out of time for audio**: it's fully decoupled — just hide/skip that tab,
  the image+video story alone is a complete, solid project.

## Honest scope note (mention this in your report, don't hide it)

The pretrained classifiers were trained on datasets that are already a bit
dated relative to the newest generative models — this is a known, documented
limitation of every public deepfake detector (see model cards). The forensic
signal fusion (ELA/FFT/symmetry) is specifically included to catch cases a
single dataset-bound classifier would miss, and is the main technical
argument for why this ensemble approach is more robust than any one model
alone.

5. **EXIF + GPS forensics**: extracts camera metadata and embedded GPS
   location (if present) from real photos — useful for investigators to
   trace when/where evidence was captured, while honestly flagging that
   some AI editors (like Google Photos) can preserve original EXIF even
   after AI-based edits.
   