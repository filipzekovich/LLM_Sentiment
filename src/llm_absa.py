from .base import ABSAAnalyzer, AspectSentiment
from .utils import extract_candidate_aspects
from typing import List
import json
import time

try:
    import ollama
except ImportError:
    ollama = None


class LLMABSA(ABSAAnalyzer):
    """
    Uses a local LLM through the Ollama Python client.
    You must have the Ollama daemon running locally (ollama serve)
    and a model (like 'llama3') downloaded via `ollama pull llama3`.
    """

    PROMPT_TEMPLATE = """
You are an assistant that extracts aspect-sentiment pairs from a sentence.
Return a JSON array of objects with keys: "aspect", "sentiment", "confidence".
Sentiment must be one of: positive, negative, neutral.
Confidence must be a decimal between 0 and 1.
Respond ONLY with the JSON array.

Examples:
Input: "The pizza was delicious but the service was terrible."
Output: [{"aspect":"pizza","sentiment":"positive","confidence":0.95},{"aspect":"service","sentiment":"negative","confidence":0.92}]

Input: "{text}"
Output:
"""

    def __init__(self, model: str = "llama3"):
        if ollama is None:
            raise RuntimeError(
                "Ollama Python client not installed. Install with `pip install ollama-python` and ensure Ollama CLI is running."
            )
        self.model = model

    def analyze(self, text: str) -> List[AspectSentiment]:
        prompt = self.PROMPT_TEMPLATE.replace("{text}", text)

        try:
            response = ollama.generate(model=self.model, prompt=prompt)

            output_text = response.get("response", "").strip()

            start = output_text.find("[")
            end = output_text.rfind("]") + 1
            if start != -1 and end != -1 and end > start:
                json_str = output_text[start:end]
                data = json.loads(json_str)
            else:
                data = json.loads(output_text)

            results = []
            for item in data:
                aspect = item.get("aspect", "")
                sentiment = item.get("sentiment", "neutral")
                confidence = float(item.get("confidence", 0.5))
                results.append(AspectSentiment(aspect, sentiment, confidence))

            return results

        except Exception as e:
            aspects = extract_candidate_aspects(text, top_n=8)
            return [AspectSentiment(a, "neutral", 0.5) for a, _ in aspects]
