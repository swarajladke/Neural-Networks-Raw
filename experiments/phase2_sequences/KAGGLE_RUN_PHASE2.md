# Executing Phase 2 Sequence Sweeps on Kaggle

This guide explains how to package the codebase and run the full Phase 2 parameter sweeps on Kaggle using a GPU or CPU accelerator.

## 1. Package the Codebase
From your local terminal, zip the project repository excluding results and cache directories:
```bash
# On Windows PowerShell
Compress-Archive -Path src, experiments, configs, tests, setup.py, README.md, ROADMAP.md -DestinationPath agnis_phase2.zip -Force
```

## 2. Upload to Kaggle
1. Go to [Kaggle](https://www.kaggle.com).
2. Create a new notebook.
3. Click "File" -> "Upload data" and upload `agnis_phase2.zip`.

## 3. Execute the Sweep inside Kaggle Notebook
Copy and execute the following cell in your Kaggle notebook:

```python
# Unzip code
!unzip -q ../input/agnis-phase2/agnis_phase2.zip -d .

# Install packages
!pip install -e .

# Run the complete sequence prediction sweep across 10 seeds and 4 conditions
!python experiments/phase2_sequences/run_sequence_sweep.py \
  --config configs/kaggle_phase2.yaml \
  --conditions periodic doublet copy palindrome \
  --models mlp_context_window simple_rnn seq_agnis_no_recurrent seq_agnis_recurrent seq_agnis_recurrent_kwta seq_agnis_recurrent_memory seq_agnis_recurrent_replay seq_agnis_full_fixed \
  --seeds 0 1 2 3 4 5 6 7 8 9

# Aggregate the seed metrics
!python experiments/phase2_sequences/summarize_phase2.py
```

## 4. Download Results
Once completed, zip and download the results directory:
```python
!zip -r results_phase2.zip results/
```
Then download `results_phase2.zip` from your Kaggle output files.
