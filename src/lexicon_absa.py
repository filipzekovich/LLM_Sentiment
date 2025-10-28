from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .base import ABSAAnalyzer, AspectSentiment
from .utils import get_spacy_model, extract_candidate_aspects, find_opinion_words_for_aspect


class LexiconABSA(ABSAAnalyzer):
    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp = get_spacy_model(spacy_model)
        self.vader = SentimentIntensityAnalyzer()

    def _score_opinion_token(self, token):
        text = token.text.lower()
        vs = self.vader.polarity_scores(text)
        compound = vs["compound"]

        if any(w in text for w in ["could", "should", "would", "maybe", "quite", "bit", "rather", "slightly"]):
            compound *= 0.4

        if "could be" in token.doc.text.lower() and compound > 0:
            compound *= -0.3

        if compound >= 0.1:
            return "positive", compound
        elif compound <= -0.1:
            return "negative", -compound
        else:
            return "neutral", 1 - abs(compound)

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        aspects = extract_candidate_aspects(text)
        results = []

        for aspect_text, (s_char, e_char) in aspects:
            aspect_token_indices = [
                i for i, token in enumerate(doc) if s_char <= token.idx < e_char
            ]
            if not aspect_token_indices:
                continue

            opinion_tokens = find_opinion_words_for_aspect(doc, aspect_token_indices)

            if not opinion_tokens:
                window = doc[max(0, aspect_token_indices[0] - 3): min(len(doc), aspect_token_indices[-1] + 4)]
                opinion_tokens = [t for t in window if t.pos_ in ("ADJ", "ADV", "VERB")]

            if not opinion_tokens:
                sent = doc[s_char:e_char].sent
                vs = self.vader.polarity_scores(sent.text)
                comp = vs["compound"]
                label = "positive" if comp >= 0.1 else "negative" if comp <= -0.1 else "neutral"
                conf = abs(comp)
                results.append(AspectSentiment(aspect_text, label, conf, (s_char, e_char)))
                continue

            pos_score, neg_score = 0.0, 0.0
            for tok in opinion_tokens:
                label, score = self._score_opinion_token(tok)

                negated = any(child.dep_ == "neg" for child in tok.children) or (
                    tok.head is not None and any(child.dep_ == "neg" for child in tok.head.children)
                )
                if negated:
                    if label == "positive":
                        label = "negative"
                    elif label == "negative":
                        label = "positive"

                if label == "positive":
                    pos_score += score
                elif label == "negative":
                    neg_score += score

            diff = pos_score - neg_score
            if diff > 0.25:
                sentiment = "positive"
                confidence = min(1.0, pos_score / (pos_score + neg_score + 1e-6))
            elif diff < -0.25:
                sentiment = "negative"
                confidence = min(1.0, neg_score / (pos_score + neg_score + 1e-6))
            else:
                sentiment = "neutral"
                confidence = 1.0 - abs(diff)

            results.append(AspectSentiment(aspect_text, sentiment, confidence, (s_char, e_char)))

        return results
