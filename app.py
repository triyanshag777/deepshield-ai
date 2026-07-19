"""
DeepShield AI - main demo app (Forensic Edition).
Run with: python app.py
Tabs: Image, Video, Audio - each with detection + a downloadable
chain-of-custody PDF report referencing applicable Indian law (BNS/IT Act).
"""

import os
import tempfile
import cv2
import gradio as gr
from PIL import Image

from src.image_detector import ImageEnsembleDetector
from src.audio_detector import AudioDetector
from src.combined_score import score_image
from src.video_detector import analyze_video
from src.saliency import occlusion_heatmap
from src.rag_explainer import RAGExplainer
from src.legal_rag import LegalReferenceAdvisor
from src.metadata_forensics import sha256_of_file, exif_report
from src.forensic_report import generate_report
from src.config import FAKE_THRESHOLD

print("=" * 60)
print("DeepShield AI — loading models (first run downloads weights)")
print("=" * 60)

image_detector = ImageEnsembleDetector()
audio_detector = AudioDetector()
rag = RAGExplainer()
legal_advisor = LegalReferenceAdvisor()

print("All models loaded. Launching UI ...")

# Keeps the last analysis result in memory so the "Generate Report" button
# doesn't need to re-run detection.
_last_image_state = {}


def analyze_image(pil_image, case_id):
    if pil_image is None:
        return "Upload an image first.", None, None, None
    result = score_image(pil_image, image_detector, return_visuals=True)
    explanation = rag.explain(result, threshold=FAKE_THRESHOLD)
    legal_note = legal_advisor.get_reference_note("image", result["final_score"], FAKE_THRESHOLD)
    exif = exif_report(pil_image)

    heatmap, _ = occlusion_heatmap(pil_image, image_detector)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    verdict = "🚨 AI-GENERATED / FAKE" if result["final_score"] >= FAKE_THRESHOLD else "✅ AUTHENTIC"
    summary = f"{verdict}   |   Confidence: {result['final_score']*100:.1f}%"

    full_explanation = explanation + "\n\n--- Metadata Forensics ---\n" + exif["suspicion_note"]
    full_explanation += "\n\n--- Legal Reference (India) ---\n" + legal_note

    _last_image_state.update({
        "pil_image": pil_image,
        "scores": result,
        "legal_note": legal_note,
        "exif_note": exif["suspicion_note"],
        "case_id": case_id or "UNSPECIFIED",
    })

    return summary, full_explanation, heatmap_rgb, "Report ready to generate below."


def generate_pdf_report(case_id):
    if not _last_image_state:
        return None
    state = _last_image_state
    tmp_dir = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, "evidence.png")
    state["pil_image"].save(img_path)
    file_hash = sha256_of_file(img_path)

    out_path = os.path.join(tmp_dir, f"deepshield_report_{case_id or 'case'}.pdf")
    generate_report(
        output_path=out_path,
        case_id=case_id or state["case_id"],
        file_name="uploaded_image.png",
        file_hash=file_hash,
        media_type="image",
        scores=state["scores"],
        legal_note=state["legal_note"],
        exif_note=state["exif_note"],
    )
    return out_path


def analyze_video_file(video_path):
    if video_path is None:
        return "Upload a video first.", None
    result = analyze_video(video_path, image_detector)
    if "error" in result:
        return result["error"], None
    explanation = rag.explain(result, threshold=FAKE_THRESHOLD)
    legal_note = legal_advisor.get_reference_note("video", result["final_score"], FAKE_THRESHOLD)
    verdict = "🚨 AI-GENERATED / FAKE" if result["final_score"] >= FAKE_THRESHOLD else "✅ AUTHENTIC"
    summary = (
        f"{verdict}   |   Confidence: {result['final_score']*100:.1f}%   "
        f"|   Frames analyzed: {result['frames_analyzed']}"
    )
    full_explanation = explanation + "\n\n--- Legal Reference (India) ---\n" + legal_note
    return summary, full_explanation


def analyze_audio_file(audio_path):
    if audio_path is None:
        return "Upload an audio file first.", None
    result = audio_detector.predict(audio_path)
    fake_prob = result["fake_prob"]
    verdict = "🚨 SYNTHETIC / AI VOICE" if fake_prob >= FAKE_THRESHOLD else "✅ AUTHENTIC VOICE"
    summary = f"{verdict}   |   Confidence: {fake_prob*100:.1f}%"
    explanation = rag.explain({"final_score": fake_prob, "model_ensemble_score": fake_prob,
                                "ela_score": None, "fft_noise_score": None, "face_found": False},
                               threshold=FAKE_THRESHOLD)
    legal_note = legal_advisor.get_reference_note("audio", fake_prob, FAKE_THRESHOLD)
    full_explanation = explanation + "\n\n--- Legal Reference (India) ---\n" + legal_note
    return summary, full_explanation


with gr.Blocks(title="DeepShield AI") as demo:
    gr.Markdown(
        "# 🛡️ DeepShield AI — Forensic Edition\n"
        "Multimodal AI-generated media detector with chain-of-custody reporting "
        "and Indian legal reference (BNS 2023 / IT Act) — built as a cyber-cell "
        "investigative triage aid."
    )

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image")
                case_id_input = gr.Textbox(label="Case ID (optional)", placeholder="e.g. CYB-2026-0417")
                img_btn = gr.Button("Analyze Image", variant="primary")
            with gr.Column():
                img_summary = gr.Textbox(label="Verdict")
                img_heatmap = gr.Image(label="Saliency Heatmap (where the model looked)")
        img_explanation = gr.Textbox(label="Forensic + Legal Explanation (RAG-grounded)", lines=14)
        report_status = gr.Textbox(label="Report Status", visible=True)
        report_btn = gr.Button("📄 Generate Chain-of-Custody PDF Report")
        report_file = gr.File(label="Download Report")

        img_btn.click(
            analyze_image,
            inputs=[img_input, case_id_input],
            outputs=[img_summary, img_explanation, img_heatmap, report_status],
        )
        report_btn.click(generate_pdf_report, inputs=case_id_input, outputs=report_file)

    with gr.Tab("Video"):
        vid_input = gr.Video(label="Upload Video")
        vid_summary = gr.Textbox(label="Verdict")
        vid_explanation = gr.Textbox(label="Forensic + Legal Explanation (RAG-grounded)", lines=12)
        vid_btn = gr.Button("Analyze Video", variant="primary")
        vid_btn.click(analyze_video_file, inputs=vid_input, outputs=[vid_summary, vid_explanation])

    with gr.Tab("Audio"):
        aud_input = gr.Audio(label="Upload Audio", type="filepath")
        aud_summary = gr.Textbox(label="Verdict")
        aud_explanation = gr.Textbox(label="Forensic + Legal Explanation (RAG-grounded)", lines=12)
        aud_btn = gr.Button("Analyze Audio", variant="primary")
        aud_btn.click(analyze_audio_file, inputs=aud_input, outputs=[aud_summary, aud_explanation])

if __name__ == "__main__":
    demo.launch()
