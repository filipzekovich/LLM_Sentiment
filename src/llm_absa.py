from .base import ABSAAnalyzer, AspectSentiment
from .utils import extract_candidate_aspects
from typing import List
import json

try:
    import ollama
except ImportError:
    ollama = None


class LLMABSA(ABSAAnalyzer):
    """
    Uses a local LLM through the Ollama Python client.
    You must have the Ollama daemon running locally (ollama serve)
    and a model (e.g., 'llama3', 'gemma3') downloaded via `ollama pull <model>`.
    """

    SYSTEM_PROMPT = """You are an assistant that extracts aspect-sentiment pairs from text.
You must return ONLY a valid JSON array with no additional text.
Each object must have: "aspect" (string), "sentiment" (positive/negative/neutral), "confidence" (0.0-1.0)."""

    FEW_SHOT_EXAMPLES = [
        {
            "input": "The pizza was delicious but the service was terrible.",
            "output": [
                {"aspect": "pizza", "sentiment": "positive", "confidence": 0.95},
                {"aspect": "service", "sentiment": "negative", "confidence": 0.92}
            ]
        },
        {
            "input": "Good battery life, but the screen could be brighter.",
            "output": [
                {"aspect": "battery life", "sentiment": "positive", "confidence": 0.88},
                {"aspect": "screen", "sentiment": "neutral", "confidence": 0.65}
            ]
        }
    ]

    def __init__(self, model: str = "gemma3:1b"):
        if ollama is None:
            raise RuntimeError(
                "Ollama Python client not installed. Install with `pip install ollama` "
                "and ensure Ollama is running with `ollama serve`."
            )
        self.model = model
        self._validate_model()

    def _validate_model(self):
        """Check if the model exists locally."""
        try:
            available_models = ollama.list()
            model_names = [m['name'].split(':')[0] for m in available_models.get('models', [])]
            if self.model not in model_names and f"{self.model}:latest" not in [m['name'] for m in available_models.get('models', [])]:
                print(f"Warning: Model '{self.model}' not found. Available models: {model_names}")
                print(f"Run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not validate model availability: {e}")

    def _build_messages(self, text: str) -> List[dict]:
        """Build the message list with system prompt and few-shot examples."""
        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT}
        ]
        
        # Add few-shot examples
        for example in self.FEW_SHOT_EXAMPLES:
            messages.append({
                'role': 'user',
                'content': f'Input: "{example["input"]}"'
            })
            messages.append({
                'role': 'assistant',
                'content': json.dumps(example["output"])
            })
        
        # Add actual query
        messages.append({
            'role': 'user',
            'content': f'Input: "{text}"'
        })
        
        return messages

    def _parse_llm_response(self, response_text: str) -> List[dict]:
        """Extract and parse JSON from LLM response."""
        # Try to find JSON array in response
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        
        if start != -1 and end > start:
            json_str = response_text[start:end]
        else:
            json_str = response_text.strip()
        
        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response_text}")

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text using local LLM to extract aspect-sentiment pairs.
        
        Falls back to neutral sentiment for extracted aspects if LLM fails.
        """
        messages = self._build_messages(text)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.3,  # Lower temperature for more consistent output
                    'num_predict': 500,  # Limit response length
                }
            )
            
            # Extract content from response
            output_text = response['message']['content'].strip()
            
            # Parse JSON response
            data = self._parse_llm_response(output_text)
            
            # Convert to AspectSentiment objects
            results = []
            for item in data:
                aspect = item.get("aspect", "").strip()
                sentiment = item.get("sentiment", "neutral").lower()
                confidence = float(item.get("confidence", 0.5))
                
                # Validate sentiment
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"
                    confidence = 0.5
                
                # Validate confidence range
                confidence = max(0.0, min(1.0, confidence))
                
                if aspect:  # Only add if aspect is not empty
                    results.append(AspectSentiment(aspect, sentiment, confidence))
            
            return results if results else self._fallback_extraction(text)
            
        except ollama.ResponseError as e:
            print(f"Ollama API error: {e.error}")
            if e.status_code == 404:
                print(f"Model '{self.model}' not found. Try: ollama pull {self.model}")
            return self._fallback_extraction(text)
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_extraction(text)

    def _fallback_extraction(self, text: str) -> List[AspectSentiment]:
        """Fallback to simple aspect extraction with neutral sentiment."""
        aspects = extract_candidate_aspects(text, top_n=5)
        return [AspectSentiment(a, "neutral", 0.5) for a, _ in aspects]
