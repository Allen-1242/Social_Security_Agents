"""
Pass 1+2: Learn the data + Build vocabulary.
Input:  POMS structured JSON (e.g. GN_02201_031.json)
Output: Enriched JSON with "vocabulary" field added to each section

For each section with text:
  1. Run spaCy noun chunks to extract terms
  2. Deduplicate via embedding cosine similarity (>= 0.65 → same concept)
  3. Add extracted terms as a "vocabulary" list on each section
  4. Build a global vocabulary across all sections
"""

import json
import re
import numpy as np
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ── Config ──────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.65
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VocabularyStore:
    """Global vocabulary with embedding-based dedup."""

    def __init__(self, model_name=EMBEDDING_MODEL, threshold=SIMILARITY_THRESHOLD):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.terms = []         # list of canonical labels
        self.embeddings = None  # cached matrix

    def _embed(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    def resolve_or_mint(self, label):
        """Returns the canonical label (deduped) for this term."""
        emb = self._embed(label)

        if self.terms:
            if self.embeddings is None:
                self.embeddings = np.stack([self._embed(t) for t in self.terms])
            sims = self.embeddings @ emb
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.threshold:
                return self.terms[best_idx]

        # New term
        self.terms.append(label)
        self.embeddings = None  # invalidate cache
        return label


def extract_vocab_from_text(text, nlp, vocab_store):
    """Extract noun chunks from text, dedup against global vocab."""
    doc = nlp(text)
    local_terms = []

    for chunk in doc.noun_chunks:
        label = chunk.text.strip()

        # Filter junk
        if len(label) < 4:
            continue
        if not re.search(r'[A-Z]', label):
            continue

        canonical = vocab_store.resolve_or_mint(label)
        if canonical not in local_terms:
            local_terms.append(canonical)

    return local_terms


def process_section(section, nlp, vocab_store):
    """
    Recursively walk a section dict:
    - If it has "text", extract vocabulary from it
    - If it has "criteria", extract from each criterion
    - If it has "note", extract from that
    - If it has "subsections", recurse into them
    """
    all_text_parts = []

    if "text" in section:
        all_text_parts.append(section["text"])

    if "note" in section:
        all_text_parts.append(section["note"])

    if "criteria" in section:
        for criterion in section["criteria"]:
            all_text_parts.append(criterion.get("text", ""))

    if "notice_contents" in section:
        for item in section["notice_contents"]:
            all_text_parts.append(item)

    # Extract vocab from all text in this section
    combined_text = " ".join(all_text_parts)
    if combined_text.strip():
        section["vocabulary"] = extract_vocab_from_text(combined_text, nlp, vocab_store)

    # Recurse into subsections
    if "subsections" in section:
        for key, subsection in section["subsections"].items():
            process_section(subsection, nlp, vocab_store)


def process_poms_json(input_path, output_path=None):
    """Main entry: load JSON, enrich with vocabulary, save."""
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading embedding model...")
    vocab_store = VocabularyStore()

    # Load input
    data = json.loads(Path(input_path).read_text())

    # Process each top-level POMS section
    for section_id, section in data.items():
        print(f"\nProcessing {section_id}...")

        if "sections" in section:
            for key, subsection in section["sections"].items():
                process_section(subsection, nlp, vocab_store)

    # Add global vocabulary summary at the top level
    for section_id in data:
        data[section_id]["global_vocabulary"] = vocab_store.terms

    # Output
    if output_path is None:
        output_path = input_path.replace(".json", "_enriched.json")

    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"\nGlobal vocabulary: {len(vocab_store.terms)} unique terms")
    print(f"Output: {output_path}")

    return data, vocab_store


if __name__ == "__main__":
    data, vocab = process_poms_json("POMS_json.json")
    print("\n── Global Vocabulary ──")
    for i, term in enumerate(vocab.terms):
        print(f"  {i+1:3d}. {term}")