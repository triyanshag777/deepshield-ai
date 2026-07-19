"""
RAG explanation module.

Retrieval-Augmented Generation, but kept fast and deterministic for a live
demo: we embed the forensic knowledge base once at startup, then at inference
time retrieve the top-k entries relevant to whichever signals actually fired,
and assemble a grounded, cited explanation. No hallucination risk because the
"generation" step is templated from retrieved text, not free-form LLM output.

(You can swap the final assembly step for a real HF text-generation pipeline
if you want free-form prose - see `generate_with_llm()` stub at the bottom.)
"""

import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

KB_PATH = Path(__file__).parent / "knowledge_base" / "forensic_indicators.md"


def _parse_kb(path):
    text = path.read_text()
    blocks = text.split("---\n")
    entries = []
    for block in blocks:
        block = block.strip()
        if not block or "id:" not in block:
            continue
        entry = {}
        m_id = re.search(r"id:\s*(\S+)", block)
        m_signal = re.search(r"signal:\s*(\S+)", block)
        m_text = re.search(r"text:\s*(.+)", block, re.DOTALL)
        if m_id and m_signal and m_text:
            entry["id"] = m_id.group(1)
            entry["signal"] = m_signal.group(1)
            entry["text"] = m_text.group(1).strip()
            entries.append(entry)
    return entries


class RAGExplainer:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("[RAGExplainer] Loading embedding model + knowledge base ...")
        self.embedder = SentenceTransformer(embed_model_name)
        self.entries = _parse_kb(KB_PATH)
        self.corpus_embeddings = self.embedder.encode(
            [e["text"] for e in self.entries], convert_to_tensor=True
        )

    def retrieve(self, query: str, top_k: int = 2, signal_filter: str = None):
        candidates = self.entries
        if signal_filter:
            candidates = [e for e in self.entries if e["signal"] == signal_filter]
            if not candidates:
                candidates = self.entries
        idxs = [self.entries.index(c) for c in candidates]
        embs = self.corpus_embeddings[idxs]
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, embs, top_k=min(top_k, len(candidates)))[0]
        return [candidates[h["corpus_id"]] for h in hits]

    def explain(self, scores: dict, threshold: float = 0.5):
        """
        scores: dict with keys like model_ensemble_score, ela_score, fft_noise_score,
                face_symmetry_score, final_score, temporal_flicker_detected (optional)
        Returns a grounded, human-readable explanation string with citations.
        """
        final = scores.get("final_score", 0.0)
        verdict = "LIKELY AI-GENERATED / MANIPULATED" if final >= threshold else "LIKELY AUTHENTIC"

        lines = [f"Verdict: {verdict}  (confidence: {final*100:.1f}%)", ""]

        # Model ensemble
        model_score = scores.get("model_ensemble_score", 0)
        query = "classifier ensemble agreement on fake image" if model_score >= threshold else "classifier ensemble uncertainty"
        hit = self.retrieve(query, top_k=1, signal_filter="model_ensemble")
        if hit:
            lines.append(f"• Classifier ensemble score: {model_score*100:.1f}% — {hit[0]['text']}")

        # ELA
        ela = scores.get("ela_score", 0)
        if ela is not None:
            query = "high compression inconsistency" if ela >= 0.5 else "uniform compression low residual"
            hit = self.retrieve(query, top_k=1, signal_filter="ela")
            if hit:
                lines.append(f"• Error Level Analysis: {ela*100:.1f}% anomaly — {hit[0]['text']}")

        # FFT
        fft = scores.get("fft_noise_score", 0)
        if fft is not None:
            query = "GAN upsampling frequency artifact" if fft >= 0.5 else "diffusion smooth spectrum"
            hit = self.retrieve(query, top_k=1, signal_filter="fft_noise")
            if hit:
                lines.append(f"• Frequency-domain (FFT) analysis: {fft*100:.1f}% anomaly — {hit[0]['text']}")

        # Face symmetry
        if scores.get("face_found"):
            sym = scores.get("face_symmetry_score", 0)
            query = "asymmetric facial features artifact" if sym >= 0.5 else "unnatural perfect symmetry"
            hit = self.retrieve(query, top_k=1, signal_filter="face_symmetry")
            if hit:
                lines.append(f"• Facial symmetry analysis: {sym*100:.1f}% anomaly — {hit[0]['text']}")

        # Temporal (video only)
        if "temporal_flicker_detected" in scores:
            if scores["temporal_flicker_detected"]:
                hit = self.retrieve("frame flicker inconsistency", top_k=1, signal_filter="temporal")
                if hit:
                    lines.append(f"• Temporal consistency: FLICKER DETECTED — {hit[0]['text']}")
            else:
                lines.append("• Temporal consistency: stable across frames (no flicker detected)")

        return "\n".join(lines)


# ---- Optional: swap in a real generative model for free-form prose ----
def generate_with_llm(retrieved_texts, scores):
    """
    Stub showing how to plug in a real HF text-generation pipeline if you want
    free-form (rather than templated) prose for the demo. Not called by default
    to keep inference fast and fully deterministic for a live judging round.

        from transformers import pipeline
        generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
        context = " ".join(retrieved_texts)
        prompt = f"Given these forensic findings: {context}\nWrite a 2-sentence summary."
        return generator(prompt, max_new_tokens=80)[0]["generated_text"]
    """
    raise NotImplementedError("Optional stretch goal - see docstring.")
