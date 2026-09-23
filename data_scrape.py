"""
Pass 1: Learn the data.
Input:  Raw PDF
Output: List of clean sentences stored as JSON corpus
"""

import re
import json
import PyPDF2
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text(pdf_path: str, start_page: int = None, end_page: int = None) -> str:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages[start_page:end_page]
        return "\n".join(page.extract_text() or "" for page in pages)

def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)       # rejoin hyphenated words
    text = re.sub(r"\n(?=[a-z])", " ", text)     # mid-sentence breaks
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def sentencize(text: str) -> list[str]:
    """Split on sentence boundaries. Preserves abbreviations like 'U.S.'"""
    # Split on period + space + capital letter
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in raw if len(s.strip()) > 10]

def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\n(?=[a-z])", " ", text)
    # Strip PDF boilerplate
    text = re.sub(r"VerDate\s+\S+.*?Jkt\s+\d+\s+PO\s+\d+.*?\n", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_corpus(pdf_path: str, output_path: str = "corpus.json", start_page: int = None, end_page: int = None) -> list[dict]:
    raw = extract_text(pdf_path, start_page, end_page)
    cleaned = clean(raw)
    sentences = sentencize(cleaned)

    corpus = [
        {"id": i, "text": s, "source": pdf_path}
        for i, s in enumerate(sentences)
    ]

    Path(output_path).write_text(json.dumps(corpus, indent=2))
    print(f"Corpus: {len(corpus)} sentences → {output_path}")
    return corpus


if __name__ == "__main__":
    build_corpus("Social_Security_Law.pdf", start_page=20, end_page=30)