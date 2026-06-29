# Phase 2 — Sequence Learning

## Goal
Test continual sequence prediction. Does Raw AGNIS retain old sequence patterns after learning new ones?

## Tasks (sequential)
- ABCABCABC... (repeating triplet)
- AABBCCAABB... (doublet repeat)
- 123123123... (numeric cycle)
- ABABAB... (simple 2-alternation)
- ABBAABBA... (palindrome-like)
- ABCDABCD... (copy 4-window)

## New Mechanisms (vs Phase 1)
- Recurrent temporal state (use_recurrent=True)
- Recurrent Hebbian update for R
- Replay of sequence prototypes during sleep phase
- Novelty detection via prediction error magnitude

## Prerequisites
- Phase 1 shows Raw AGNIS forgetting < Naive MLP forgetting ✓

## Status: 🔲 Planned — begins after Phase 1 success
