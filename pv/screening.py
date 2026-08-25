from typing import Callable


LABEL_RELEVANT = "reports a drug safety issue, adverse event, or manufacturing defect"
LABEL_NOT_RELEVANT = "does not relate to drug safety"


def screen_article(article: dict, classifier=None) -> dict:
    text = f"{article.get('title', '')}. {article.get('abstract', '') or ''}".strip()
    if not text.strip(". "):
        return {"relevant": True, "confidence": 0.0, "reason": "no text to classify"}
    if classifier is None:
        return {"relevant": True, "confidence": 0.0, "reason": "classifier unavailable; manual review required"}
    result = classifier(text, [LABEL_RELEVANT, LABEL_NOT_RELEVANT], multi_label=False)
    label, score = result["labels"][0], float(result["scores"][0])
    return {
        "relevant": label == LABEL_RELEVANT,
        "confidence": round(score, 3),
        "reason": f"zero-shot top label: '{label}' (score={score:.2f})",
    }


def load_classifier():
    from transformers import pipeline
    import torch
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
    )


def screen_batch(articles: list[dict], threshold: float = 0.6,
                 classifier=None, progress: Callable[[int, int], None] | None = None) -> dict[str, list[dict]]:
    buckets = {"relevant": [], "not_relevant": [], "needs_review": []}
    for index, source in enumerate(articles, start=1):
        article = dict(source)
        result = screen_article(article, classifier)
        article["screening"] = result
        if result["confidence"] < threshold:
            buckets["needs_review"].append(article)
        elif result["relevant"]:
            buckets["relevant"].append(article)
        else:
            buckets["not_relevant"].append(article)
        if progress:
            progress(index, len(articles))
    return buckets
