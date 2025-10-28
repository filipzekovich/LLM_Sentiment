from .base import ABSAAnalyzer, AspectSentiment
from .utils import get_spacy_model, extract_candidate_aspects
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List
import torch

class ML_ABSA(ABSAAnalyzer):
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1", device=None):
        self.model_name = model_name
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # create a pipeline to simplify prediction
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device, return_all_scores=True)

    def analyze(self, text: str) -> List[AspectSentiment]:
        aspects = extract_candidate_aspects(text, top_n=12)
        results = []
        for aspect_text, (s_char, e_char) in aspects:
            model_input = f"{text} [SEP] {aspect_text}"
            try:
                preds = self.pipe(model_input)
                scores = {p['label'].lower(): p['score'] for p in preds[0]}
                best_label = max(scores.items(), key=lambda kv: kv[1])[0]
                confidence = float(scores[best_label])
            except Exception as e:
                best_label, confidence = "neutral", 0.5
            results.append(AspectSentiment(aspect_text, best_label, confidence, (s_char, e_char)))
        return results
