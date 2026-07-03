"""
Raw AGNIS — src/agnis/text/__init__.py
"""

from agnis.text.text_domains import (
    TextDomain,
    ProseDomain,
    CodeDomain,
    ArithmeticDomain,
    DialogueDomain,
    get_all_domains
)
from agnis.text.char_pipeline import CharVocab, CharacterStream
from agnis.text.char_metrics import CharMetrics, compute_forgetting, compute_bpc_forgetting, compute_growth_efficiency

# Phase 5 story domains
from agnis.text.story_domains import (
    StoryDomain,
    AnimalsDomain,
    ObjectsDomain,
    EmotionsDomain,
    ActionsDomain,
    get_all_story_domains
)
from agnis.text.conditional_generation import generate_continuation
from agnis.text.generation_metrics import (
    compute_repetition_rate,
    compute_keyword_retention,
    compute_name_consistency,
    compute_sentence_completion,
    compute_distinct_n
)
