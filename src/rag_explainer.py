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

    def get_by_id(self, entry_id: str):
        """Direct lookup by exact indicator ID."""
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

    def explain(self, scores: dict, threshold: float = 0.5):
        final = scores.get("final_score", 0.0)
        verdict = "LIKELY AI-GENERATED / MANIPULATED" if final >= threshold else "LIKELY AUTHENTIC"

        lines = [f"Verdict: {verdict}  (confidence: {final*100:.1f}%)", ""]

        model_score = scores.get("model_ensemble_score", 0)
        hit = self.get_by_id("model_ensemble_agree" if model_score >= threshold else "model_ensemble_disagree")
        if hit:
            lines.append(f"• Classifier ensemble score: {model_score*100:.1f}% — {hit['text']}")

        ela = scores.get("ela_score", 0)
        if ela is not None:
            hit = self.get_by_id("ela_high" if ela >= 0.5 else "ela_low")
            if hit:
                lines.append(f"• Error Level Analysis: {ela*100:.1f}% anomaly — {hit['text']}")

        fft = scores.get("fft_noise_score", 0)
        if fft is not None:
            hit = self.get_by_id("fft_high" if fft >= 0.5 else "fft_diffusion")
            if hit:
                lines.append(f"• Frequency-domain (FFT) analysis: {fft*100:.1f}% anomaly — {hit['text']}")

        if scores.get("face_found"):
            sym = scores.get("face_symmetry_score", 0)
            hit = self.get_by_id("face_symmetry_high" if sym >= 0.5 else "face_symmetry_low")
            if hit:
                lines.append(f"• Facial symmetry analysis: {sym*100:.1f}% anomaly — {hit['text']}")

        if "temporal_flicker_detected" in scores:
            if scores["temporal_flicker_detected"]:
                hit = self.get_by_id("temporal_flicker")
                if hit:
                    lines.append(f"• Temporal consistency: FLICKER DETECTED — {hit['text']}")
            else:
                lines.append("• Temporal consistency: stable across frames (no flicker detected)")

        return "\n".join(lines)


def generate_with_llm(retrieved_texts, scores):
    """Optional stub - not used by default."""
    raise NotImplementedError("Optional stretch goal - see docstring.")
