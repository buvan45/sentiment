import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List


class FinBertSentimentAnalyzer:
    """
    Wrapper around FinBERT (financial BERT) for sentiment analysis.

    Uses the HuggingFace model: yiyanghkust/finbert-tone
    which outputs: positive, neutral, negative.
    """

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        self.model_name = model_name

        # Load tokenizer & model
        print(f"Loading FinBERT model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Map label indices to their names, e.g. {0: 'neutral', 1: 'positive', 2: 'negative'}
        self.id2label = self.model.config.id2label

    def _predict_single(self, text: str) -> Dict:
        """Run FinBERT on a single text and return sentiment details."""

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits and convert to probabilities
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        # Build a mapping from label name to probability
        label_probs = {}
        for idx, prob in enumerate(probs):
            label = self.id2label[idx].lower()  # e.g., "positive", "neutral", "negative"
            label_probs[label] = float(prob)

        # Some models might use different label names; handle robustly
        positive = label_probs.get("positive", 0.0)
        negative = label_probs.get("negative", 0.0)
        neutral = label_probs.get("neutral", 0.0)

        # Sentiment score: positive - negative (range approx -1 to +1)
        sentiment_score = positive - negative

        # Pick overall label by max probability
        if positive >= negative and positive >= neutral:
            overall_label = "positive"
        elif negative >= positive and negative >= neutral:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        return {
            "text": text,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "score": sentiment_score,
            "label": overall_label,
        }

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Run FinBERT on a list of texts.

        Returns a list of dicts with keys:
        - text
        - positive
        - neutral
        - negative
        - score  (positive - negative)
        - label  ("positive"/"negative"/"neutral")
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for t in texts:
            if not t or not t.strip():
                continue
            result = self._predict_single(t.strip())
            results.append(result)

        return results


def demo():
    """Quick test for the FinBERT sentiment analyzer."""
    analyzer = FinBertSentimentAnalyzer()

    sample_headlines = [
        "Tesla beats earnings expectations and shares surge in after-hours trading",
        "Government launches investigation into major bank for fraud allegations",
        "Infosys signs multi-billion dollar contract with a US healthcare giant",
        "Company reports mixed quarterly results with flat revenue growth",
    ]

    print("\n=== FinBERT Sentiment Demo ===\n")
    results = analyzer.predict(sample_headlines)

    for res in results:
        print(f"Text     : {res['text']}")
        print(f"Label    : {res['label']}")
        print(f"Score    : {res['score']:.4f}  (pos={res['positive']:.4f}, neg={res['negative']:.4f}, neu={res['neutral']:.4f})")
        print("-" * 80)


if __name__ == "__main__":
    demo()
