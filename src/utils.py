import spacy
from typing import List, Tuple


_nlp = None
def get_spacy_model(name="en_core_web_sm"):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(name)
    return _nlp

def extract_candidate_aspects(text: str, top_n=10) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Extract candidate aspect terms from text using noun chunks and nouns.
    Returns list of (aspect_text, (start_char, end_char))
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    seen = set()
    aspects = []
    # prefer noun_chunks (multiword aspects)
    for nc in doc.noun_chunks:
        term = nc.text.strip()
        if term.lower() not in seen:
            seen.add(term.lower())
            aspects.append((term, (nc.start_char, nc.end_char)))
    # fallback: single nouns
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            term = token.text.strip()
            if term.lower() not in seen:
                seen.add(term.lower())
                aspects.append((term, (token.idx, token.idx + len(token.text))))
    return aspects[:top_n]

def find_opinion_words_for_aspect(doc, aspect_token_indices):
    """
    Given a spaCy doc and the token indices for an aspect phrase,
    return nearby opinion words (adjectives/verbs/adverbs) via dependency relations.
    """
    opinion_tokens = []
    aspect_tokens = [doc[i] for i in aspect_token_indices]
    for tok in aspect_tokens:
        for child in tok.children:
            if child.pos_ in ("ADJ", "ADV"):
                opinion_tokens.append(child)
        head = tok.head
        if head is not None and head.pos_ in ("ADJ", "VERB", "ADV"):
            opinion_tokens.append(head)
    return list({t.i: t for t in opinion_tokens}.values())