from typing import Callable



LABEL_RELEVANT = "reports a drug safety issue, adverse event, or manufacturing defect"

LABEL_NOT_RELEVANT = "does not relate to drug safety"

DEFAULT_BATCH_SIZE = 4

DEFAULT_MAX_TEXT_CHARS = 6000





def _article_text(article: dict, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    
    text = f"{article.get('title', '')}. {article.get('abstract', '') or ''}".strip()
    
    return text[:max_chars]
    




def _result_from_model(result: dict) -> dict:
    
    label, score = result["labels"][0], float(result["scores"][0])
    
    return {
        
        "relevant": label == LABEL_RELEVANT,
        
        "confidence": round(score, 3),
        
        "reason": f"zero-shot top label: '{label}' (score={score:.2f})",
        
    }
    




def screen_article(article: dict, classifier=None) -> dict:
    
    text = _article_text(article)
    
    if not text.strip(". "):
        
        return {"relevant": True, "confidence": 0.0, "reason": "no text to classify"}
        
    if classifier is None:
        
        return {
            
            "relevant": True,
            
            "confidence": 0.0,
            
            "reason": "classifier unavailable; manual review required",
            
        }
        
    result = classifier(
        
        text,
        
        [LABEL_RELEVANT, LABEL_NOT_RELEVANT],
        
        multi_label=False,
        
        truncation=True,
        
    )
    
    return _result_from_model(result)
    




def load_classifier():
    
    from transformers import pipeline
    
    import torch
    


    return pipeline(
        
        "zero-shot-classification",
        
        model="facebook/bart-large-mnli",
        
        device=0 if torch.cuda.is_available() else -1,
        
    )
    




def _empty_buckets() -> dict[str, list[dict]]:
    
    return {"relevant": [], "not_relevant": [], "needs_review": []}
    




def _add_to_bucket(
    
    buckets: dict[str, list[dict]],
    
    article: dict,
    
    result: dict,
    
    threshold: float,
    
) -> None:
    
    article["screening"] = result
    
    if result["confidence"] < threshold:
        
        buckets["needs_review"].append(article)
        
    elif result["relevant"]:
        
        buckets["relevant"].append(article)
        
    else:
        
        buckets["not_relevant"].append(article)
        




def screen_batch(
    
    articles: list[dict],
    
    threshold: float = 0.6,
    
    classifier=None,
    
    progress: Callable[[int, int], None] | None = None,
    
    batch_size: int = DEFAULT_BATCH_SIZE,
    
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    
) -> dict[str, list[dict]]:
    
    """Screen articles in small batches to reduce transformer overhead."""
    
    buckets = _empty_buckets()
    
    total = len(articles)
    
    if total == 0:
        
        return buckets
        


    if classifier is None:
        
        for index, source in enumerate(articles, start=1):
            
            article = dict(source)
            
            _add_to_bucket(
                
                buckets,
                
                article,
                
                {
                    
                    "relevant": True,
                    
                    "confidence": 0.0,
                    
                    "reason": "classifier unavailable; manual review required",
                    
                },
                
                threshold,
                
            )
            
            if progress:
                
                progress(index, total)
                
        return buckets
        


    safe_batch_size = max(1, min(int(batch_size), 16))
    
    labels = [LABEL_RELEVANT, LABEL_NOT_RELEVANT]
    
    pending_articles: list[dict] = []
    
    pending_texts: list[str] = []
    


    def flush_pending() -> None:
        
        if not pending_articles:
            
            return
            
        outputs = classifier(
            
            pending_texts,
            
            labels,
            
            multi_label=False,
            
            batch_size=safe_batch_size,
            
            truncation=True,
            
        )
        
        if isinstance(outputs, dict):
            
            outputs = [outputs]
            
        for article, output in zip(pending_articles, outputs):
            
            _add_to_bucket(buckets, article, _result_from_model(output), threshold)
            
        pending_articles.clear()
        
        pending_texts.clear()
        


    completed = 0
    
    for source in articles:
        
        article = dict(source)
        
        text = _article_text(article, max_text_chars)
        
        if not text.strip(". "):
            
            _add_to_bucket(
                
                buckets,
                
                article,
                
                {"relevant": True, "confidence": 0.0, "reason": "no text to classify"},
                
                threshold,
                
            )
            
            completed += 1
            
            if progress:
                
                progress(completed, total)
                
            continue
            


        pending_articles.append(article)
        
        pending_texts.append(text)
        
        if len(pending_articles) >= safe_batch_size:
            
            flush_pending()
            
            completed += safe_batch_size
            
            if progress:
                
                progress(completed, total)
                


    remaining = len(pending_articles)
    
    flush_pending()
    
    if remaining:
        
        completed += remaining
        
        if progress:
            
            progress(completed, total)
            


    return buckets
    
































































































































