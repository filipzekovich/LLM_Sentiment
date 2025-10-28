# Aspect-Based Sentiment Analysis (ABSA) Project

A comprehensive implementation of Aspect-Based Sentiment Analysis using three different approaches: Lexicon-based, Transformer-based, and LLM-based methods.

## Overview

This project extracts aspects (features, attributes) from text and determines the sentiment expressed toward each aspect. Unlike traditional sentiment analysis that provides overall sentiment, ABSA identifies specific aspects and their associated sentiments.

**Example:**
- Input: `"The pizza was delicious but the service was terrible."`
- Output:
  - Aspect: "pizza" → Sentiment: POSITIVE
  - Aspect: "service" → Sentiment: NEGATIVE

## Data

The evaluation/test data used in this project is a small curated sample derived from the "Amazon Reviews for Dog Food Product" dataset (Kaggle): https://www.kaggle.com/datasets/unwrangle/amazon-reviews-for-dog-food-product. A simplified subset is included in data/amazon_dog_food_reviews.json.
## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Ollama installed for LLM-based approach

### Step 1: Clone or Download the Project

```bash
cd /path/to/project
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```


### Step 4: Setup Ollama for LLM Implementation

1. Install Ollama from https://ollama.ai
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull model and run (in a new terminal):
   ```bash
   ollama pull gemma3:1b

   ollama run gemma3:1b
   ```

## Usage

### Quick Start

```python
from src import LexiconABSA, ML_ABSA, LLMABSA

text = "The pizza was delicious but the service was terrible."

# Method 1: Lexicon-Based
lexicon_analyzer = LexiconABSA()
results = lexicon_analyzer.analyze(text)
for r in results:
    print(f"{r.aspect}: {r.sentiment} ({r.confidence:.2f})")

# Method 2: Transformer-Based
ml_analyzer = ML_ABSA()
results = ml_analyzer.analyze(text)
for r in results:
    print(f"{r.aspect}: {r.sentiment} ({r.confidence:.2f})")

# Method 3: LLM-Based (requires Ollama)
llm_analyzer = LLMABSA(model="gemma3:1b")
results = llm_analyzer.analyze(text)
for r in results:
    print(f"{r.aspect}: {r.sentiment} ({r.confidence:.2f})")
```

### Implementation 1: Lexicon-Based ABSA

Uses spaCy for linguistic analysis and VADER for sentiment scoring.

```python
from src import LexiconABSA

analyzer = LexiconABSA(spacy_model="en_core_web_sm")
results = analyzer.analyze("Great battery life but poor camera quality.")

# Results: List[AspectSentiment]
# Each AspectSentiment has:
#   - aspect: str (e.g., "battery life")
#   - sentiment: str ("positive", "negative", or "neutral")
#   - confidence: float (0.0 to 1.0)
#   - text_span: tuple (start_char, end_char)
```

**Features:**
- Extracts noun chunks and nouns as aspect candidates
- Uses dependency parsing to find opinion words
- Handles negations (e.g., "not good")
- Applies VADER lexicon for sentiment scoring

### Implementation 2: Transformer-Based ABSA

Uses pre-trained DeBERTa model fine-tuned for ABSA.

```python
from src import ML_ABSA

# Automatically uses GPU if available
analyzer = ML_ABSA(model_name="yangheng/deberta-v3-base-absa-v1.1")
results = analyzer.analyze("The screen is bright and beautiful.")

# Can specify device explicitly:
# analyzer = ML_ABSA(device=0)  # GPU 0
# analyzer = ML_ABSA(device=-1) # CPU
```

**Features:**
- Uses state-of-the-art transformer model
- Extracts aspects using spaCy, classifies sentiment using DeBERTa
- Returns confidence scores from model probabilities
- Handles complex sentiment expressions

### Implementation 3: LLM-Based ABSA

Uses local LLM through Ollama with few-shot prompting.

```python
from src import LLMABSA

# Use smaller model for speed
analyzer = LLMABSA(model="gemma3:1b")

results = analyzer.analyze("Food was excellent, service was slow.")

```

**Features:**
- Few-shot prompting with examples
- JSON-structured output parsing
- Fallback mechanism if LLM unavailable
- Temperature control for consistent results

## API Reference

### Base Classes

```python
@dataclass
class AspectSentiment:
    aspect: str              # The aspect mentioned (e.g., "pizza")
    sentiment: str           # "positive", "negative", or "neutral"
    confidence: float        # Confidence score 0.0 to 1.0
    text_span: Tuple[int, int] = None  # Character positions in original text

class ABSAAnalyzer:
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and return aspect-sentiment pairs."""
        raise NotImplementedError
```

### Implementations

All three implementations follow the same interface:

```python
analyzer = LexiconABSA()    # or ML_ABSA() or LLMABSA()
results = analyzer.analyze(text)  # Returns List[AspectSentiment]
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_absa.py

# Skip LLM tests if Ollama not available
pytest -v -k "not llm"
```

## Design Decisions

### 1. Lexicon-Based Approach
- **Choice:** VADER lexicon with spaCy dependency parsing
- **Rationale:** VADER is pre-tuned for social media text and handles intensifiers/negations well
- **Aspect Extraction:** Noun chunks preferred over single nouns for multi-word aspects

### 2. Transformer Approach
- **Choice:** DeBERTa-v3 fine-tuned for ABSA
- **Rationale:** State-of-the-art performance on ABSA benchmarks
- **Input Format:** `[TEXT] [SEP] [ASPECT]` for aspect-specific sentiment classification

### 3. LLM Approach
- **Choice:** Local Ollama with Gemma3 model
- **Rationale:** Privacy, no API costs, reproducible results, small model size
- **Prompting:** Few-shot examples + JSON output format for structured extraction

### 4. Unified API
- Consistent interface across all implementations
- Easy to swap implementations for comparison
- Standardized output format



## Examples

See `notebooks/comparison.ipynb` for detailed examples and comparison of all three methods.

Quick verification:
```bash
python tests/check.py
```