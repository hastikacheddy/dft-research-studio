"""
data/pdf_processor.py
---------------------
Stateless PDF ingestion utilities for the DFT corpus.
"""

from __future__ import annotations

import re
from typing import Dict

import fitz  # PyMuPDF


class ScientificPDFProcessor:
    """Extracts and sections text from scientific PDFs."""

    # Regex that identifies common scientific section headers
    _SECTION_PATTERN = re.compile(
        r"(^\s*(?:[IVXLCDM]+\.|\d+\.|[A-Z]\.)?\s*"
        r"(?:ABSTRACT|INTRODUCTION|METHODS|RESULTS|DISCUSSION|"
        r"CONCLUSION|REFERENCES|ACKNOWLEDG(?:E)?MENTS|"
        r"SUPPORTING INFORMATION|COMPUTATIONAL DETAILS)"
        r"(?:\s[A-Z]+)*\s*$)",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text normalisation for PDF extracts."""
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)   # fix hyphenation
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
        text = re.sub(r"\n{2,}", "\n\n", text)              # excess newlines
        text = re.sub(r"\s+", " ", text).strip()            # excess spaces
        return text

    @staticmethod
    def clean_citations(text: str) -> str:
        """Remove inline citations such as [1], [1-3], (Smith, 2020)."""
        text = re.sub(r"\[[\d, -]+\]", "", text)
        text = re.sub(r"\([\d\w, -]+\)", "", text)
        return text

    @classmethod
    def extract_sections(cls, pdf_path: str) -> Dict[str, str]:
        """
        Parse a PDF with PyMuPDF, skip header/footer margins, and split the
        text into logical sections (Abstract, Methods, …).

        Returns
        -------
        dict  {section_name: section_text}  — empty dict on read failure.
        """
        import os

        raw_text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                rect = page.rect
                # Clip to skip top 8 % (header) and bottom 8 % (footer)
                clip = fitz.Rect(
                    rect.x0,
                    rect.y0 + rect.height * 0.08,
                    rect.x1,
                    rect.y1 - rect.height * 0.08,
                )
                raw_text += page.get_text("text", clip=clip) + "\n"
            doc.close()
        except Exception as exc:
            print(f"[PDFProcessor] Error reading {os.path.basename(pdf_path)}: {exc}")
            return {}

        parts = cls._SECTION_PATTERN.split(raw_text)

        if len(parts) <= 1:
            # Fallback: treat entire document as one chunk
            return {"CONTENT": cls.clean_text(raw_text)}

        sections: Dict[str, str] = {
            "HEADER": cls.clean_text(parts[0])
        }
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = cls.clean_text(parts[i]).upper()
                body = cls.clean_citations(cls.clean_text(parts[i + 1]))
                sections[header] = body

        return sections
