"""
Phase 5 Story Domains tests.
"""

from agnis.text.story_domains import (
    AnimalsDomain,
    ObjectsDomain,
    EmotionsDomain,
    ActionsDomain,
    get_all_story_domains
)
from agnis.text import CharVocab


def test_story_generation_basic():
    domains = get_all_story_domains()
    assert len(domains) == 4
    
    for d in domains:
        stories = d.generate_stories(n_stories=5, seed=42)
        assert len(stories) == 5
        for s in stories:
            assert "prompt" in s
            assert "target" in s
            assert len(s["prompt"]) > 0
            assert len(s["target"]) > 0
            assert s["target"].endswith("\n")


def test_story_generation_determinism():
    d = AnimalsDomain()
    stories_1 = d.generate_stories(n_stories=3, seed=42)
    stories_2 = d.generate_stories(n_stories=3, seed=42)
    stories_3 = d.generate_stories(n_stories=3, seed=43)
    
    assert stories_1 == stories_2
    assert stories_1 != stories_3


def test_story_characters_vocab():
    vocab = CharVocab()
    for d in get_all_story_domains():
        stories = d.generate_stories(n_stories=10, seed=10)
        for s in stories:
            full_text = s["prompt"] + s["target"]
            for char in full_text:
                assert char in vocab.char_to_idx, f"Char {repr(char)} not in default vocab"
