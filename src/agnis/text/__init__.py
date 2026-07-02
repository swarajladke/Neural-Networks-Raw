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
