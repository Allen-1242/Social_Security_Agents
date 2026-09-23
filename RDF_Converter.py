"""
Pass 3: Triplify (A-Box population) — Vocab-Grounded.
Input:  corpus.json + VocabularyStore from pass 2
Output: triples.ttl — RDF triples grounded in the T-Box vocabulary

For every sentence:
  1. Retrieve top-k relevant vocab entries (RAG over T-Box)
  2. Inject retrieved entities + relations into LLM prompt
  3. LLM extracts (s, p, o) triples constrained to known vocabulary
  4. Strict resolution: exact match first, then high-threshold embedding fallback
  5. Emit RDF triples with PROV provenance
"""

import json
import re
import torch
import numpy as np
from pathlib import Path

from rdflib import Graph, Literal, URIRef, Namespace, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, PROV

from Vocabulary import VocabularyStore, build_vocabulary, get_extractor, SSA, GOV


# ── Vocab-Grounded Triple Extraction ───────────────────────────────────

import numpy as np

TRIPLE_SYSTEM_TEMPLATE = """You are a knowledge graph triple extractor. Given a sentence and a VOCABULARY of known entities and relations, extract all (subject, predicate, object) triples.

CRITICAL: You MUST use labels from the vocabulary below. Do NOT invent new entities or relations.
If a concept in the sentence matches a vocabulary item, use that vocabulary label exactly.
If nothing in the vocabulary matches, skip that triple.

=== AVAILABLE ENTITIES ===
{entities}

=== AVAILABLE RELATIONS ===
{relations}

Return ONLY valid JSON. No markdown, no preamble.

{{
  "triples": [
    {{"subject": "label from AVAILABLE ENTITIES", "predicate": "label from AVAILABLE RELATIONS", "object": "label from AVAILABLE ENTITIES"}}
  ]
}}

If no triples can be formed from the vocabulary, return {{"triples": []}}
ONLY return JSON."""


def retrieve_relevant_vocab(
    sentence: str,
    vocab: VocabularyStore,
    top_k: int = 15,
    min_sim: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """
    RAG over the T-Box: embed the sentence, retrieve the top-k most
    relevant vocab entries by cosine similarity.

    Returns (entities, relations) — each a list of
    {"label": ..., "uri": ..., "sim": ...}
    """
    sent_emb = vocab._embed(sentence)
    sims = vocab._cosine_similarities(sent_emb)

    if len(sims) == 0:
        return [], []

    # Get top-k indices above minimum similarity
    ranked_idxs = np.argsort(sims)[::-1]

    entities = []
    relations = []

    for idx in ranked_idxs[:top_k]:
        sim = float(sims[idx])
        if sim < min_sim:
            break

        entry = vocab.entries[idx]
        item = {
            "label": entry.canonical_label,
            "uri": entry.uri,
            "sim": sim,
            "alt_labels": entry.alt_labels,
        }

        if entry.entry_type == "relation":
            relations.append(item)
        else:
            entities.append(item)

    return entities, relations


def format_vocab_for_prompt(entities: list[dict], relations: list[dict]) -> str:
    """Format retrieved vocab entries into the prompt template."""
    ent_lines = []
    for e in entities:
        line = f"- {e['label']}"
        if e["alt_labels"]:
            line += f"  (also known as: {', '.join(e['alt_labels'])})"
        ent_lines.append(line)

    rel_lines = []
    for r in relations:
        line = f"- {r['label']}"
        if r["alt_labels"]:
            line += f"  (also known as: {', '.join(r['alt_labels'])})"
        rel_lines.append(line)

    return TRIPLE_SYSTEM_TEMPLATE.format(
        entities="\n".join(ent_lines) if ent_lines else "(none retrieved)",
        relations="\n".join(rel_lines) if rel_lines else "(none retrieved)",
    )


def resolve_label_strict(label: str, vocab: VocabularyStore) -> str | None:
    """
    Strict resolution: the LLM was given exact labels, so we first try
    exact match, then fall back to embedding similarity.
    """
    label_clean = label.strip()

    # 1. Exact match on canonical or alt labels
    for entry in vocab.entries:
        if label_clean == entry.canonical_label:
            return entry.uri
        if label_clean in entry.alt_labels:
            return entry.uri

    # 2. Fuzzy fallback via embedding (handles minor LLM rewording)
    emb = vocab._embed(label_clean)
    sims = vocab._cosine_similarities(emb)

    if len(sims) > 0:
        best_idx = int(sims.argmax())
        best_sim = sims[best_idx]
        # Higher threshold here — the LLM was told to use exact labels,
        # so any match should be very close
        if best_sim >= 0.90:
            return vocab.entries[best_idx].uri

    return None


def extract_triples_llm(sentence: str, vocab: VocabularyStore) -> list[dict]:
    """
    Vocab-grounded triple extraction:
    1. Retrieve relevant vocab entries for this sentence
    2. Inject them into the LLM prompt
    3. LLM maps sentence content to known vocabulary
    """
    extractor = get_extractor()

    # Retrieve relevant vocabulary
    entities, relations = retrieve_relevant_vocab(sentence, vocab)

    # Build grounded prompt
    system_prompt = format_vocab_for_prompt(entities, relations)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
    text = extractor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = extractor.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=extractor.tokenizer.eos_token_id,
        )

    response = extractor.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = extractor._parse_response(response)

    if "triples" not in parsed:
        return []
    return parsed["triples"]


def triplify(corpus: list[dict], vocab: VocabularyStore) -> Graph:
    """Walk corpus, retrieve relevant vocab, extract grounded triples, emit RDF."""

    g = Graph()
    g.bind("ssa", SSA)
    g.bind("gov", GOV)
    g.bind("prov", PROV)

    triple_count = 0
    skipped = 0

    for sent in corpus:
        text = sent["text"]
        sent_uri = SSA[f"sentence_{sent['id']}"]

        raw_triples = extract_triples_llm(text, vocab)

        for raw in raw_triples:
            subj_text = raw.get("subject", "")
            rel_text = raw.get("predicate", "")
            obj_text = raw.get("object", "")

            # Strict resolution — LLM was given exact labels
            subj_uri = resolve_label_strict(subj_text, vocab)
            rel_uri = resolve_label_strict(rel_text, vocab)
            obj_uri = resolve_label_strict(obj_text, vocab)

            if subj_uri and obj_uri:
                s = URIRef(subj_uri)
                o = URIRef(obj_uri)

                if rel_uri:
                    p = URIRef(rel_uri)
                else:
                    p = SSA[re.sub(r"[^a-zA-Z0-9]+", "_", rel_text.lower())]

                g.add((s, p, o))

                # Provenance
                reification = BNode()
                g.add((reification, RDF.type, RDF.Statement))
                g.add((reification, RDF.subject, s))
                g.add((reification, RDF.predicate, p))
                g.add((reification, RDF.object, o))
                g.add((reification, PROV.wasDerivedFrom, sent_uri))
                g.add((sent_uri, RDF.type, PROV.Entity))
                g.add((sent_uri, RDFS.label, Literal(text)))

                triple_count += 1
                print(f"  TRIPLE: <{subj_text}> {rel_text} <{obj_text}>")
            else:
                skipped += 1

    print(f"\nA-Box: {triple_count} triples extracted, {skipped} skipped (unresolved)")
    return g


# ── Main ────────────────────────────────────────────────────────────────
def run_pipeline(
    corpus_path: str = "corpus.json",
    vocab_path: str = "vocabulary.ttl",
    output_path: str = "triples.ttl",
):
    # Load corpus
    corpus = json.loads(Path(corpus_path).read_text())

    # Build or load vocabulary (pass 2)
    vocab = build_vocabulary(corpus_path, vocab_path)

    # Triplify (pass 3)
    g = triplify(corpus, vocab)

    # Merge T-Box into output for a complete graph
    tbox = Graph()
    tbox.parse(vocab_path, format="turtle")
    combined = tbox + g

    combined.serialize(output_path, format="turtle")
    print(f"\nComplete graph: {len(combined)} triples → {output_path}")

    # Also emit JSON-LD for downstream consumption
    combined.serialize(output_path.replace(".ttl", ".jsonld"), format="json-ld")


if __name__ == "__main__":
    run_pipeline()