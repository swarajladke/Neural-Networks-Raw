"""
Raw AGNIS — src/agnis/text/generation_metrics.py

Evaluation metrics for character-level conditional story generation quality.
"""

from typing import List, Dict, Set

# Predefined domain keywords for validation
DOMAIN_KEYWORDS = {
    "animals": {"cat", "dog", "bird", "frog", "bear", "owl", "fox", "named", "liked", "park", "forest", "garden", "lake", "saw", "ant", "bee", "fish", "mouse", "together", "happy"},
    "objects": {"toy", "ball", "box", "key", "book", "cup", "owned", "lost", "yard", "room", "bag", "table", "looked", "found", "smiled"},
    "emotions": {"sad", "happy", "scared", "brave", "child", "walked", "park", "lake", "woods", "school", "beach", "found", "coin", "shell", "stone", "flower", "leaf", "feel", "glad", "proud", "excited", "calm", "good"},
    "actions": {"run", "jump", "play", "help", "climb", "swim", "morning", "park", "pool", "tree", "hill", "path", "ran", "jumped", "played", "helped", "climbed", "swam", "bench", "tent", "gate", "fence", "fun"}
}

ALL_NAMES = {"mimi", "toby", "pip", "hops", "bruno", "hoot", "reddy", "sam", "ann", "leo", "mia", "ben", "joy"}


def compute_repetition_rate(text: str, n: int = 3) -> float:
    """Compute the n-gram repetition rate in generated text.
    
    repetition_rate = 1.0 - (unique_ngrams / total_ngrams)
    """
    if len(text) < n:
        return 0.0
        
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
        
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))


def compute_keyword_retention(text: str, domain: str) -> float:
    """Compute the fraction of domain keywords present in the generated text."""
    if domain not in DOMAIN_KEYWORDS:
        return 0.0
        
    keywords = DOMAIN_KEYWORDS[domain]
    words = set(text.lower().split())
    
    # Strip basic punctuation from words
    cleaned_words = set()
    for w in words:
        cleaned = "".join(c for c in w if c.isalnum())
        if cleaned:
            cleaned_words.add(cleaned)
            
    matches = keywords.intersection(cleaned_words)
    return len(matches) / len(keywords)


def compute_name_consistency(text: str) -> float:
    """Verify if the first character name introduced is consistently used.
    
    Returns 1.0 if the name appears at least twice in the text, 0.0 if it appears
    only once, and 0.5 if no names from the vocabulary are found.
    """
    words = [w.strip(".,;:!?\n") for w in text.lower().split()]
    found_names = [w for w in words if w in ALL_NAMES]
    
    if not found_names:
        return 0.5
        
    first_name = found_names[0]
    count = words.count(first_name)
    
    return 1.0 if count >= 2 else 0.0


def compute_sentence_completion(text: str) -> float:
    """Check if the text ends with valid terminal punctuation (. ! ? or trailing newline)."""
    trimmed = text.strip()
    if not trimmed:
        return 0.0
    return 1.0 if trimmed[-1] in {".", "!", "?"} else 0.0
