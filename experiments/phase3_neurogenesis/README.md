# Phase 3 — Neurogenesis

## Goal
Add autonomous structural growth. Demonstrate that unit birth reduces persistent prediction error and improves continual learning without uncontrolled growth.

## New Mechanisms (vs Phase 2)
- Growth score computation (G_l)
- Unit birth with error-residual initialization
- Maturity gate (new units earn influence by reducing error)
- Usage tracking, importance tracking, redundancy detection
- Pruning: remove low-usage, low-importance, high-redundancy units
- Merging: combine nearly-identical units

## Prerequisites
- Phase 2 shows Raw AGNIS sequence forgetting < Simple RNN ✓

## Status: 🔲 Planned — begins after Phase 2 success
