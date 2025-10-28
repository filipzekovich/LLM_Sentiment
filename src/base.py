from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    confidence: float
    text_span: Tuple[int, int] = None

class ABSAAnalyzer:

    def analyze(self, text: str) -> List[AspectSentiment]:
        raise NotImplementedError("analyze must be implemented by subclasses")