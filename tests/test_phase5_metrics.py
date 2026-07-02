"""
Phase 5 Metrics validation tests.
"""

from agnis.text.generation_metrics import (
    compute_repetition_rate,
    compute_keyword_retention,
    compute_name_consistency,
    compute_sentence_completion
)


def test_repetition_rate():
    # Repeats "abc" 3 times
    text = "abc abc abc"
    # trigrams: "abc", "bc ", "c a", " ab", "abc", "bc ", "c a", " ab", "abc"
    # Total trigrams = 11 - 3 + 1 = 9
    # set(trigrams) = {"abc", "bc ", "c a", " ab"} (size 4)
    # repetition_rate = 1.0 - 4/9 = 5/9 = 0.555
    rep = compute_repetition_rate(text, n=3)
    assert 0.5 < rep < 0.6
    
    # Completely unique
    assert compute_repetition_rate("abcdefgh", n=3) == 0.0


def test_keyword_retention():
    # Domain: animals. Keywords include cat, dog, park, happy...
    text = "the cat was happy in the park"
    ret = compute_keyword_retention(text, "animals")
    # Matches: cat, happy, park (3 matches out of 20 keywords)
    assert ret == 3 / 20
    
    # Non-existent domain
    assert compute_keyword_retention(text, "unknown") == 0.0


def test_name_consistency():
    # Introduced name "mimi", appears twice
    text1 = "once there was a cat named mimi. mimi was happy."
    assert compute_name_consistency(text1) == 1.0
    
    # Introduced "mimi", appears once
    text2 = "once there was a cat named mimi."
    assert compute_name_consistency(text2) == 0.0
    
    # No names present from vocabulary
    text3 = "once there was a cat."
    assert compute_name_consistency(text3) == 0.5


def test_sentence_completion():
    assert compute_sentence_completion("the story is done.") == 1.0
    assert compute_sentence_completion("the story is done!") == 1.0
    assert compute_sentence_completion("is the story done?") == 1.0
    assert compute_sentence_completion("incomplete story") == 0.0
    assert compute_sentence_completion("") == 0.0
