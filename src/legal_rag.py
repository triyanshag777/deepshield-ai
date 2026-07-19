"""
Legal-RAG module: retrieves applicable BNS 2023 / IT Act sections based on
the media type and detection outcome. Kept as a *reference note for
investigators*, not a legal determination — always shown with a disclaimer.

This is what differentiates DeepShield from a plain classifier project: it
bridges "this is fake" to "here is what an investigator can act on."
"""

import re
from pathlib import Path

KB_PATH = Path(__file__).parent / "knowledge_base" / "legal_sections.md"

DISCLAIMER = (
    "NOTE: This is an automated triage reference, not legal advice. "
    "Applicability of any section depends on case-specific facts and must be "
    "confirmed by a legal professional or the investigating officer. "
    "Complaints can also be filed at cybercrime.gov.in."
)


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
        m_applies = re.search(r"applies_to:\s*\[([^\]]+)\]", block)
        m_cond = re.search(r"condition:\s*(\S+)", block)
        m_text = re.search(r"text:\s*(.+)", block, re.DOTALL)
        if m_id and m_applies and m_cond and m_text:
            entry["id"] = m_id.group(1)
            entry["applies_to"] = [x.strip() for x in m_applies.group(1).split(",")]
            entry["condition"] = m_cond.group(1)
            entry["text"] = m_text.group(1).strip()
            entries.append(entry)
    return entries


class LegalReferenceAdvisor:
    def __init__(self):
        self.entries = _parse_kb(KB_PATH)

    def get_reference_note(self, media_type: str, fake_confidence: float, threshold: float = 0.5):
        """
        media_type: "image" | "video" | "audio"
        fake_confidence: 0-1 float from the detector
        Returns a formatted reference note string.
        """
        condition = "high_confidence_fake" if fake_confidence >= threshold else None
        applicable = [
            e for e in self.entries
            if media_type in e["applies_to"]
            and (e["condition"] == "any" or e["condition"] == condition)
        ]

        if not applicable:
            return "No specific statutory reference triggered at this confidence level.\n\n" + DISCLAIMER

        lines = [f"Potentially applicable provisions ({media_type}, confidence {fake_confidence*100:.1f}%):", ""]
        for e in applicable:
            lines.append(f"• {e['text']}")
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)
