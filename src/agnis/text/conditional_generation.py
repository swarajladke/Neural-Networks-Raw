"""
Raw AGNIS — src/agnis/text/conditional_generation.py

Prompt conditioning and character-level story generation decoding routines.
"""

import torch
import torch.nn.functional as F
from agnis.text.char_pipeline import CharVocab
from agnis.sequence.sequence_wrapper import SequenceModel

def generate_continuation(
    model: SequenceModel,
    prompt: str,
    vocab: CharVocab,
    max_chars: int = 160,
    decoding: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 5,
    stop_on_double_newline: bool = True,
    repetition_penalty: float = 1.0
) -> str:
    """Condition the model on the prompt and generate a text continuation.
    
    Ensures that evaluation mode is strictly followed: no weights, memory,
    importance trackers, or capacity dimensions are updated. Only the
    transient recurrent hidden state is stepped.
    """
    model.reset_sequence_state()
    
    # 1. Condition on prompt transitions
    for i in range(len(prompt) - 1):
        x_char = prompt[i]
        y_char = prompt[i + 1]
        x_oh = vocab.to_onehot(x_char)
        y_oh = vocab.to_onehot(y_char)
        model.advance_state_only(x_oh, y_oh)
        
    # 2. Begin generation from the last prompt token
    generated_text = ""
    current_char = prompt[-1]
    
    # Track recent history for repetition penalty (simple penalty for recently generated chars)
    generated_chars = []
    
    for _ in range(max_chars):
        current_oh = vocab.to_onehot(current_char)
        
        # Predict next char distribution without state update
        pred_probs = model.predict_no_state_update(current_oh)
        
        # Apply repetition penalty if specified and greater than 1.0
        if repetition_penalty > 1.0 and len(generated_chars) > 0:
            # Shift probabilities down for recently generated characters
            # pred_probs is a probability distribution (sum = 1.0)
            # We can convert back to logits, apply penalty, and softmax again
            eps = 1e-8
            logits = torch.log(pred_probs + eps)
            for rc in set(generated_chars[-5:]):  # penalize characters from the last 5 steps
                idx = vocab.encode(rc)
                # If logit is positive, divide by penalty. If negative, multiply.
                if logits[idx] > 0:
                    logits[idx] /= repetition_penalty
                else:
                    logits[idx] *= repetition_penalty
            pred_probs = torch.softmax(logits, dim=-1)

        # Decode token
        if decoding == "greedy" or temperature == 0.0:
            next_idx = pred_probs.argmax().item()
        else:
            # Temperature scaling
            eps = 1e-8
            logits = torch.log(pred_probs + eps) / max(temperature, eps)
            
            # Top-k filtering
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, vocab.vocab_size))
                min_value = values[-1]
                logits[logits < min_value] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
        next_char = vocab.decode(next_idx)
        generated_text += next_char
        generated_chars.append(next_char)
        
        # Advance recurrent state
        next_oh = vocab.to_onehot(next_char)
        model.advance_state_only(current_oh, next_oh)
        
        current_char = next_char
        
        # Early stopping checks
        if stop_on_double_newline and generated_text.endswith("\n\n"):
            break
            
    return generated_text
