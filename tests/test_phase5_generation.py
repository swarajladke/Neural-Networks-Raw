"""
Phase 5 Conditional Generation and Evaluation Leakage tests.
"""

import torch
from agnis.text import CharVocab, CharacterStream
from agnis.text.story_domains import AnimalsDomain
from agnis.text.conditional_generation import generate_continuation
from agnis.sequence.sequence_wrapper import SeqAgnisModel
from agnis.utils.config import load_config


def test_generation_no_capacity_change():
    """Verify that generation does not change weights, capacity, births, prunes, or memory size."""
    vocab = CharVocab()
    config = load_config("configs/phase5_smoke.yaml")
    
    model = SeqAgnisModel(
        d_in=vocab.vocab_size,
        d_out=vocab.vocab_size,
        d_z=16,
        config=config,
        max_latent_dim=64
    )
    
    # Train model on a few characters to populate weights and memory
    d = AnimalsDomain()
    story = d.generate_stories(n_stories=1, seed=42)[0]
    prompt = story["prompt"]
    target = story["target"]
    
    for x_oh, y_oh, _, _ in CharacterStream(prompt + target, vocab):
        model.train_transition(x_oh, y_oh)
        
    # Snapshot parameters before generation
    D_before = model.base_model.cell.D.clone()
    E_before = model.base_model.cell.E.clone()
    R_before = model.base_model.cell.R.clone()
    d_z_before = model.base_model.cell.d_z
    mem_size_before = model.base_model.fast_mem.size if model.base_model.fast_mem else 0
    
    # Generate continuation
    gen_text = generate_continuation(
        model=model,
        prompt=prompt,
        vocab=vocab,
        max_chars=40,
        decoding="greedy"
    )
    
    # Verify exact equality of parameter weights
    assert torch.equal(model.base_model.cell.D, D_before), "D weights mutated during generation!"
    assert torch.equal(model.base_model.cell.E, E_before), "E weights mutated during generation!"
    assert torch.equal(model.base_model.cell.R, R_before), "R weights mutated during generation!"
    
    # Verify structural metrics are unchanged
    assert model.base_model.cell.d_z == d_z_before, "Recurrent latent capacity mutated during generation!"
    
    mem_size_after = model.base_model.fast_mem.size if model.base_model.fast_mem else 0
    assert mem_size_after == mem_size_before, "FastMemory size changed during generation!"
    
    assert len(gen_text) > 0, "No continuation generated"
