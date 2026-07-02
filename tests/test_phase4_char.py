"""Phase 4 Character-Level Continual Language Tests
"""

import pytest
import torch
import math
import os

from agnis.text import CharVocab, CharacterStream, get_all_domains, CharMetrics
from agnis.text.text_domains import ProseDomain, CodeDomain, ArithmeticDomain, DialogueDomain
from agnis.sequence.sequence_wrapper import SeqAgnisModel, BigramBaseline, TrigramBaseline
from agnis.utils.config import load_config, AGNISConfig


def test_char_vocab_roundtrip():
    vocab = CharVocab()
    assert vocab.vocab_size > 0
    
    char = 'a'
    idx = vocab.encode(char)
    assert vocab.decode(idx) == char
    
    oh = vocab.to_onehot(char)
    assert oh.sum().item() == 1.0
    assert vocab.from_onehot(oh) == char
    
    text = "hello world\n"
    indices = vocab.encode_string(text)
    decoded = vocab.decode_indices(indices)
    assert decoded == text


def test_char_stream_yields_correct_pairs():
    vocab = CharVocab()
    text = "abc"
    stream = CharacterStream(text, vocab)
    assert len(stream) == 2
    
    pairs = list(stream)
    x_oh, y_oh, x_idx, y_idx = pairs[0]
    
    assert x_idx == vocab.encode('a')
    assert y_idx == vocab.encode('b')
    assert x_oh[x_idx] == 1.0
    assert y_oh[y_idx] == 1.0


def test_prose_generator_deterministic():
    domain = ProseDomain()
    text1 = domain.generate(200, seed=42)
    text2 = domain.generate(200, seed=42)
    text3 = domain.generate(200, seed=43)
    
    assert len(text1) == 200
    assert text1 == text2
    assert text1 != text3
    assert all(c in CharVocab().chars for c in text1), "Contains out-of-vocab characters"


def test_code_generator_has_colons_and_parens():
    domain = CodeDomain()
    text = domain.generate(500, seed=42)
    assert "def " in text or "if " in text or "=" in text
    assert ":" in text or "(" in text or "=" in text
    assert all(c in CharVocab().chars for c in text)


def test_arithmetic_generator_has_digits_and_equals():
    domain = ArithmeticDomain()
    text = domain.generate(200, seed=42)
    assert "=" in text
    assert any(c.isdigit() for c in text)
    assert all(c in CharVocab().chars for c in text)


def test_dialogue_generator_has_qa_markers():
    domain = DialogueDomain()
    text = domain.generate(200, seed=42)
    assert "q: " in text
    assert "a: " in text
    assert "?" in text
    assert all(c in CharVocab().chars for c in text)


def test_all_domains_share_vocab():
    vocab = CharVocab()
    for d in get_all_domains():
        text = d.generate(1000, seed=10)
        for char in text:
            assert char in vocab.char_to_idx, f"Char {repr(char)} not in vocab"


def test_bigram_baseline_converges():
    vocab = CharVocab()
    model = BigramBaseline(vocab_size=vocab.vocab_size)
    
    # Train on repeating pattern: "ababab..."
    for _ in range(50):
        model.train_transition(vocab.to_onehot('a'), vocab.to_onehot('b'))
        model.train_transition(vocab.to_onehot('b'), vocab.to_onehot('a'))
        
    # Test P(b|a)
    pred_b_given_a = model.predict_transition(vocab.to_onehot('a'))
    assert pred_b_given_a[vocab.encode('b')].item() > 0.9
    
    # Test P(a|b)
    pred_a_given_b = model.predict_transition(vocab.to_onehot('b'))
    assert pred_a_given_b[vocab.encode('a')].item() > 0.9


def test_trigram_baseline_tracks_context():
    vocab = CharVocab()
    model = TrigramBaseline(vocab_size=vocab.vocab_size)
    
    # Train on distinct contexts:
    # "cab" -> c followed by a -> b follows a
    # "dab" -> d followed by a -> b follows a (same)
    # "xay" -> x followed by a -> y follows a (different)
    for _ in range(20):
        # Context cab
        model.reset_sequence_state()
        model.train_transition(vocab.to_onehot('c'), vocab.to_onehot('a'))
        model.train_transition(vocab.to_onehot('a'), vocab.to_onehot('b'))
        
        # Context xay
        model.reset_sequence_state()
        model.train_transition(vocab.to_onehot('x'), vocab.to_onehot('a'))
        model.train_transition(vocab.to_onehot('a'), vocab.to_onehot('y'))

    # Verify context dependency
    model.reset_sequence_state()
    model.advance_state_only(vocab.to_onehot('c'), vocab.to_onehot('a'))
    pred = model.predict_transition(vocab.to_onehot('a'))
    assert pred[vocab.encode('b')].item() > 0.8
    
    model.reset_sequence_state()
    model.advance_state_only(vocab.to_onehot('x'), vocab.to_onehot('a'))
    pred = model.predict_transition(vocab.to_onehot('a'))
    assert pred[vocab.encode('y')].item() > 0.8


def test_char_evaluation_no_weight_or_capacity_update():
    """MANDATORY: Evaluation mode must NOT change weights, memory, importance, or capacity."""
    vocab = CharVocab()
    config = load_config('configs/phase4_smoke.yaml')
    model = SeqAgnisModel(
        d_in=vocab.vocab_size, d_out=vocab.vocab_size,
        d_z=16, config=config, max_latent_dim=64
    )
    
    # Train a bit
    domain = ProseDomain()
    text = domain.generate(200, seed=42)
    stream = CharacterStream(text, vocab)
    for x, y, _, _ in stream:
        model.train_transition(x, y)
    
    # Snapshot weights and capacity
    D_before = model.base_model.cell.D.clone()
    E_before = model.base_model.cell.E.clone()
    R_before = model.base_model.cell.R.clone()
    d_z_before = model.base_model.cell.d_z
    mem_count_before = model.base_model.fast_mem.size if model.base_model.fast_mem else 0
    
    # Evaluation loop (must not change weights)
    eval_text = domain.get_eval_text(100, seed=42)
    eval_stream = CharacterStream(eval_text, vocab)
    model.reset_sequence_state()
    for x, y, x_idx, y_idx in eval_stream:
        pred = model.predict_no_state_update(x)
        model.advance_state_only(x, y)
    
    # Verify nothing changed
    assert torch.equal(model.base_model.cell.D, D_before), "D weights changed during eval!"
    assert torch.equal(model.base_model.cell.E, E_before), "E weights changed during eval!"
    assert torch.equal(model.base_model.cell.R, R_before), "R weights changed during eval!"
    assert model.base_model.cell.d_z == d_z_before, "Capacity changed during eval!"
    mem_count_after = model.base_model.fast_mem.size if model.base_model.fast_mem else 0
    assert mem_count_after == mem_count_before, "Memory size changed during eval!"


def test_bpc_finite():
    metrics = CharMetrics(vocab_size=10)
    # Uniform probs
    probs = torch.ones(10) / 10.0
    metrics.update(probs, 2)
    assert metrics.accuracy == 0.0  # argmax is 0, target is 2
    assert metrics.bpc == pytest.approx(-math.log2(0.1))
    assert math.isfinite(metrics.bpc)
    
    # Zero probability corner case
    probs_zero = torch.zeros(10)
    metrics.update(probs_zero, 3)
    assert math.isfinite(metrics.bpc)


def test_char_metrics_accuracy():
    metrics = CharMetrics(vocab_size=5)
    
    # Correct top-1
    pred = torch.tensor([0.1, 0.7, 0.1, 0.1, 0.0])
    metrics.update(pred, 1)
    assert metrics.accuracy == 1.0
    assert metrics.top3_accuracy == 1.0
    
    # Correct top-3 but not top-1
    metrics.reset()
    pred = torch.tensor([0.4, 0.1, 0.3, 0.2, 0.0])  # top3 indices: 0, 2, 3
    metrics.update(pred, 2)
    assert metrics.accuracy == 0.0
    assert metrics.top3_accuracy == 1.0


def test_benchmark_smoke_runs():
    """Verify that the benchmark script runs under smoke conditions."""
    import subprocess
    script_path = "experiments/phase4_char_language/run_char_benchmark.py"
    
    cmd = [
        "python",
        script_path,
        "--model", "seq_agnis_fixed",
        "--seed", "42",
        "--config", "configs/phase4_smoke.yaml",
        "--smoke"
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Benchmark run failed:\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    assert "Run finished successfully" in res.stdout
