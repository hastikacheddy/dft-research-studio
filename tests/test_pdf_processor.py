"""
tests/test_pdf_processor.py
-----------------------------
Tests for ScientificPDFProcessor text-cleaning and section-splitting logic.

All assertions are made against the actual regex and string-manipulation
algorithms in the module.  No PDF files are required on disk; the section
splitter is tested by calling it directly on strings that match the exact
pattern it uses on real documents.
"""

from __future__ import annotations

import pytest
from dft_research_studio.data.pdf_processor import ScientificPDFProcessor


# ─────────────────────────────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────────────────────────────

class TestCleanText:
    """
    Verified against real PyMuPDF output artefacts encountered in DFT papers.
    """

    def test_dehyphenates_broken_words(self):
        # Common in two-column PDF layouts where "den-\nsity" spans columns
        result = ScientificPDFProcessor.clean_text("den-\nsity functional")
        assert result == "density functional"

    def test_replaces_fi_ligature(self):
        result = ScientificPDFProcessor.clean_text("ﬁrst")
        assert result == "first"

    def test_replaces_fl_ligature(self):
        result = ScientificPDFProcessor.clean_text("ﬂow")
        assert result == "flow"

    def test_collapses_multiple_blank_lines_to_two(self):
        raw = "paragraph one\n\n\n\nparagraph two"
        result = ScientificPDFProcessor.clean_text(raw)
        assert "\n\n\n" not in result
        assert "paragraph one" in result and "paragraph two" in result

    def test_collapses_internal_whitespace(self):
        result = ScientificPDFProcessor.clean_text("The  electron    density")
        assert "  " not in result

    def test_strips_leading_and_trailing_whitespace(self):
        result = ScientificPDFProcessor.clean_text("   DFT calculation   ")
        assert result == "DFT calculation"

    def test_empty_string_returns_empty(self):
        assert ScientificPDFProcessor.clean_text("") == ""

    def test_idempotent_on_clean_text(self):
        clean = "The exchange-correlation functional."
        assert ScientificPDFProcessor.clean_text(clean) == clean

    def test_multiple_artefacts_corrected_in_one_pass(self):
        raw = "ﬁrst-\nprinciples  cal-\nculation"
        result = ScientificPDFProcessor.clean_text(raw)
        assert "first" in result
        assert "calculation" in result
        assert "  " not in result


# ─────────────────────────────────────────────────────────────────────
# clean_citations
# ─────────────────────────────────────────────────────────────────────

class TestCleanCitations:
    """
    Citation formats are taken from real DFT papers in the corpus.
    The assertion is that scientific content is preserved while
    citation markers are removed.
    """

    def test_removes_single_numeric_citation(self):
        text = "as reported previously [1]"
        result = ScientificPDFProcessor.clean_citations(text)
        assert "[1]" not in result
        assert "as reported previously" in result

    def test_removes_citation_range(self):
        result = ScientificPDFProcessor.clean_citations("see refs [12-15]")
        assert "[12-15]" not in result

    def test_removes_multiple_citations_in_brackets(self):
        result = ScientificPDFProcessor.clean_citations("results [1, 3, 7]")
        assert "[1, 3, 7]" not in result

    def test_removes_parenthetical_citation(self):
        result = ScientificPDFProcessor.clean_citations("(Grimme, 2010)")
        assert "(Grimme, 2010)" not in result

    def test_preserves_chemical_formulas_with_subscripts(self):
        # "H2O" must not be removed — it matches no citation pattern
        text = "The water molecule H2O has angle 104.5 degrees"
        result = ScientificPDFProcessor.clean_citations(text)
        assert "H2O" in result

    def test_preserves_numerical_values(self):
        text = "The MAE is 0.23 kcal/mol [5]"
        result = ScientificPDFProcessor.clean_citations(text)
        assert "0.23" in result
        assert "kcal/mol" in result

    def test_no_op_on_citation_free_text(self):
        text = "The PBE0 functional performs well on non-covalent interactions."
        assert ScientificPDFProcessor.clean_citations(text) == text


# ─────────────────────────────────────────────────────────────────────
# Section regex (internal _SECTION_PATTERN)
# ─────────────────────────────────────────────────────────────────────

class TestSectionPattern:
    """
    Directly tests the compiled regex against strings representative of
    real section headers encountered in the 25-paper DFT corpus.
    """

    _PATTERN = ScientificPDFProcessor._SECTION_PATTERN

    def _matches(self, line: str) -> bool:
        return bool(self._PATTERN.search(line + "\n"))

    def test_matches_abstract(self):
        assert self._matches("ABSTRACT")

    def test_matches_introduction(self):
        assert self._matches("INTRODUCTION")

    def test_matches_methods(self):
        assert self._matches("METHODS")

    def test_matches_results(self):
        assert self._matches("RESULTS")

    def test_matches_conclusion(self):
        assert self._matches("CONCLUSION")

    def test_matches_references(self):
        assert self._matches("REFERENCES")

    def test_matches_computational_details(self):
        assert self._matches("COMPUTATIONAL DETAILS")

    def test_matches_numbered_section(self):
        # e.g. "2. METHODS" — common in ACS journal format
        assert self._matches("2. METHODS")

    def test_matches_roman_numeral_section(self):
        assert self._matches("II. RESULTS")

    def test_does_not_match_body_sentence(self):
        assert not self._matches(
            "The exchange-correlation functional used in this study is PBE0."
        )

    def test_does_not_match_partial_word(self):
        # "introduction" embedded in a sentence should not split sections
        assert not self._matches(
            "As an introduction to density functional theory, we note that"
        )
