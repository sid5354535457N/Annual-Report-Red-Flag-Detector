"""PDF text extraction with conservative numeric-artifact handling.

The parser preserves the existing ``page`` and ``text`` interface used by the
application. If it encounters a known numeric glyph-mapping problem or a
suspicious digit/letter combination, it adds ``raw_text`` and
``extraction_warnings`` to that page so the original extracted text remains
available for audit. It never guesses unknown digits.
"""

from __future__ import annotations

import re

import pymupdf


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be processed correctly."""


# In the supplied TCS annual report, the numeral 4 is extracted as U+03F0
# GREEK KAPPA SYMBOL (ϰ) in numeric strings such as ``75ϰ`` and ``202ϰ``.
# Convert this known, context-specific glyph only when it appears in a token
# that also contains an ordinary digit. Keep the raw page text and emit a
# warning whenever this normalization is applied.
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?:\d[\d,ϰ]*(?:\.[\dϰ]+)?|ϰ[\d,ϰ]*(?:\.[\dϰ]+)?)"
)
_KNOWN_NUMERIC_GLYPHS = {"ϰ": "4"}

# Heuristic only: these combinations are not changed automatically. They are
# surfaced for review, because a letter mixed into a number can be genuine
# notation (for example, 5G) or a damaged digit (for example, 202x).
_MIXED_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z])\d[\d,.]*[A-Za-z][A-Za-z0-9,.]*"
)
_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def _normalize_known_numeric_glyphs(text: str) -> tuple[str, list[str]]:
    """Normalize one known numeric glyph mapping and return audit warnings."""
    warnings: list[str] = []

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if "ϰ" not in token or not any(char.isdigit() for char in token):
            return token

        normalized = "".join(_KNOWN_NUMERIC_GLYPHS.get(char, char) for char in token)
        warnings.append(
            "Known PDF font/glyph mapping encountered in numeric-looking text: "
            f"{token!r} was normalized to {normalized!r}. Verify material figures "
            "against the original PDF page image before relying on them."
        )
        return normalized

    return _NUMERIC_TOKEN_PATTERN.sub(replace_token, text), warnings


def _find_suspicious_numeric_tokens(text: str) -> list[str]:
    """Return likely digit/letter extraction artifacts without changing them."""
    found: list[str] = []
    seen: set[str] = set()

    for match in _MIXED_NUMERIC_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token.casefold().endswith(_ORDINAL_SUFFIXES):
            continue
        # A digit joined to a letter is only a warning. Do not alter it, since
        # the token may be legitimate (e.g. a technical identifier).
        if token not in seen:
            found.append(token)
            seen.add(token)

    return found


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract selectable text from a PDF, preserving PDF page numbers.

    Args:
        pdf_bytes: The PDF file contents as bytes.

    Returns:
        A list of page dictionaries. Every page includes ``page`` (1-based)
        and ``text``. If a numeric extraction artifact is found, that page
        also includes ``raw_text`` (text exactly as PyMuPDF extracted it) and
        ``extraction_warnings`` (warnings requiring source-image review).

    Raises:
        PDFExtractionError: If the PDF is invalid, empty, password-protected,
        or contains no extractable text.
    """
    if not pdf_bytes:
        raise PDFExtractionError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise PDFExtractionError(
                    "This PDF is password-protected. "
                    "Please upload an accessible copy."
                )

            pages: list[dict] = []

            # Page numbers displayed to the user start at 1 and refer to the
            # PDF viewer page index, not the printed page label in the report.
            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text("text").strip()
                text, normalization_warnings = _normalize_known_numeric_glyphs(
                    raw_text
                )

                warnings = list(normalization_warnings)
                suspicious_tokens = _find_suspicious_numeric_tokens(raw_text)
                for token in suspicious_tokens:
                    # The known glyph case is already covered by the more
                    # specific normalization warning above.
                    if "ϰ" in token:
                        continue
                    warnings.append(
                        f"Suspicious digit/letter combination {token!r} found "
                        "in extracted text. It was not changed automatically; "
                        "verify it against the original PDF page image."
                    )

                page_record = {"page": page_number, "text": text}
                if warnings:
                    page_record["raw_text"] = raw_text
                    page_record["extraction_warnings"] = warnings
                pages.append(page_record)

    except PDFExtractionError:
        raise
    except Exception as error:
        raise PDFExtractionError(
            "Could not read this file as a valid PDF."
        ) from error

    if not pages:
        raise PDFExtractionError("The PDF contains no pages.")

    # Scanned PDFs may contain page images rather than selectable text.
    if not any(page["text"] for page in pages):
        raise PDFExtractionError(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR support will be added in a later step."
        )

    return pages
