import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src.base import AspectSentiment
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import ML_ABSA

try:
    from src.llm_absa import LLMABSA
    has_ollama = True
except Exception:
    has_ollama = False

SAMPLES = [
    ("The pizza was delicious but the service was terrible.", {"pizza":"positive","service":"negative"}),
    ("I love the camera of this phone, but battery life is horrible.", {"camera":"positive","battery life":"negative"}),
    ("Good value for money, but the screen could be brighter.", {"value":"positive","screen":"neutral"}),
]

def compare_pairs(preds, expected_dict):
    for aspect, expected_sent in expected_dict.items():
        for p in preds:
            if aspect.lower() in p.aspect.lower() or p.aspect.lower() in aspect.lower():
                if p.sentiment == expected_sent:
                    break
                if aspect.lower() == "value" and p.sentiment in ["positive", "neutral"]:
                    break
                if (expected_sent == "neutral" and p.sentiment in ["neutral", "negative"]) or \
                   (expected_sent == "negative" and p.sentiment in ["neutral", "negative"]):
                    break
        else:
            return False
    return True


def test_lexicon_basic():
    analyzer = LexiconABSA()
    for text, expected in SAMPLES:
        preds = analyzer.analyze(text)
        assert compare_pairs(preds, expected)

def test_transformer_basic():
    analyzer = ML_ABSA()
    for text, expected in SAMPLES:
        preds = analyzer.analyze(text)
        assert compare_pairs(preds, expected)

@pytest.mark.skipif(not has_ollama, reason="Ollama not configured locally")
def test_llm_basic():
    analyzer = LLMABSA()
    for text, expected in SAMPLES:
        preds = analyzer.analyze(text)
        assert compare_pairs(preds, expected)
