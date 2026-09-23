"""
Pass 2: Build vocabulary (T-Box).
Input:  corpus.json from pass 1
Output: vocabulary.ttl — deduplicated OWL vocabulary

Pipeline:
  1. spaCy NER extracts entities from each sentence
  2. Embed each entity with sentence-transformers
  3. Cosine similarity >= 0.85 → same concept (dedup)
  4. Below threshold → mint new URI
"""

import json
import re
import numpy as np
import spacy
from pathlib import Path
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS


# ── Config ──────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SSA = Namespace("http://example.org/ssa/")
GOV = Namespace("http://example.org/gov/")


# ── Embedding Store ─────────────────────────────────────────────────────
@dataclass
class VocabEntry:
    uri: str
    canonical_label: str
    alt_labels: list[str] = field(default_factory=list)
    entry_type: str = "entity"  # "entity" | "relation" | "class"
    embedding: np.ndarray = None


class VocabularyStore:
    """In-memory vocabulary with embedding-based dedup."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, threshold: float = SIMILARITY_THRESHOLD):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.entries: list[VocabEntry] = []
        self._embeddings_matrix: np.ndarray | None = None

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def _cosine_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        if not self.entries:
            return np.array([])
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.stack([e.embedding for e in self.entries])
        return self._embeddings_matrix @ query_emb

    def resolve_or_mint(self, label: str, entry_type: str = "entity", namespace: Namespace = SSA) -> str:
        emb = self._embed(label)
        sims = self._cosine_similarities(emb)

        if len(sims) > 0:
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            if best_sim >= self.threshold:
                existing = self.entries[best_idx]
                if label not in existing.alt_labels and label != existing.canonical_label:
                    existing.alt_labels.append(label)
                    print(f"  DEDUP [{best_sim:.3f}]: '{label}' → '{existing.canonical_label}'")
                return existing.uri

        slug = self._slugify(label)
        uri = str(namespace[slug])
        entry = VocabEntry(
            uri=uri,
            canonical_label=label,
            entry_type=entry_type,
            embedding=emb,
        )
        self.entries.append(entry)
        self._embeddings_matrix = None
        print(f"  NEW {entry_type}: '{label}' → <{uri}>")
        return uri

    @staticmethod
    def _slugify(name: str) -> str:
        s = re.sub(r"\(.*?\)", "", name).strip()
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
        return s

    def to_owl(self, output_path: str = "vocabulary.ttl") -> Graph:
        g = Graph()
        g.bind("ssa", SSA)
        g.bind("gov", GOV)
        g.bind("owl", OWL)
        g.bind("skos", SKOS)

        ont_uri = URIRef("http://example.org/ssa/ontology")
        g.add((ont_uri, RDF.type, OWL.Ontology))
        g.add((ont_uri, RDFS.label, Literal("Social Security Act Ontology")))

        for entry in self.entries:
            uri = URIRef(entry.uri)

            if entry.entry_type == "class":
                g.add((uri, RDF.type, OWL.Class))
            elif entry.entry_type == "relation":
                g.add((uri, RDF.type, OWL.ObjectProperty))
            else:
                g.add((uri, RDF.type, OWL.NamedIndividual))

            g.add((uri, RDFS.label, Literal(entry.canonical_label)))
            g.add((uri, SKOS.prefLabel, Literal(entry.canonical_label)))

            for alt in entry.alt_labels:
                g.add((uri, SKOS.altLabel, Literal(alt)))

        g.serialize(output_path, format="turtle")
        print(f"\nT-Box: {len(self.entries)} concepts → {output_path}")
        self.print_vocab()
        return g

    def print_vocab(self):
        print("\n" + "=" * 60)
        print("VOCABULARY SUMMARY")
        print("=" * 60)
        for entry in self.entries:
            tag = {"entity": "ENT", "relation": "REL", "class": "CLS"}[entry.entry_type]
            print(f"\n  [{tag}] {entry.canonical_label}")
            print(f"        URI: {entry.uri}")
            if entry.alt_labels:
                print(f"        aka: {', '.join(entry.alt_labels)}")
        print("=" * 60)


# ── NER-based Term Extraction ───────────────────────────────────────────

# spaCy NER label → vocab entry type + namespace
NER_TYPE_MAP = {
    "ORG":      ("entity", GOV),
    "GPE":      ("entity", GOV),
    "LAW":      ("class", SSA),
    "PERSON":   ("entity", GOV),
    "FAC":      ("entity", GOV),
    "NORP":     ("entity", GOV),
}


SIMILARITY_THRESHOLD = 0.65

def extract_terms(corpus: list[dict], vocab: VocabularyStore) -> VocabularyStore:
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("Extracting noun chunks...\n")

    for sent in corpus:
        doc = nlp(sent["text"])

        for chunk in doc.noun_chunks:
            label = chunk.text.strip()

            if len(label) < 4:
                continue
            if not re.search(r'[A-Z]', label):
                continue

            vocab.resolve_or_mint(label, entry_type="entity", namespace=GOV)

    return vocab

def is_valid_entity(label: str) -> bool:
    if len(label) < 4:
        return False
    # Must contain at least one uppercase word
    if not re.search(r'[A-Z][a-z]{2,}', label):
        return False
    # Reject lone abbreviations
    if re.match(r'^[A-Z\.]{1,4}$', label):
        return False
    return True


# ── Main ────────────────────────────────────────────────────────────────
def build_vocabulary(corpus_path: str = "corpus.json", output_path: str = "vocabulary.ttl") -> VocabularyStore:
    corpus = json.loads(Path(corpus_path).read_text())
    vocab = VocabularyStore()
    vocab = extract_terms(corpus, vocab)
    vocab.to_owl(output_path)
    return vocab


if __name__ == "__main__":
    vocab = build_vocabulary()
    print(f"\nVocabulary stats:")
    print(f"  Entities:  {sum(1 for e in vocab.entries if e.entry_type == 'entity')}")
    print(f"  Relations: {sum(1 for e in vocab.entries if e.entry_type == 'relation')}")
    print(f"  Classes:   {sum(1 for e in vocab.entries if e.entry_type == 'class')}")
