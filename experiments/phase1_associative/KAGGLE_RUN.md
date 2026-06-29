# Running Raw AGNIS Benchmark Sweeps on Kaggle

To avoid heavy local execution times, the Raw AGNIS `v0.2` multi-seed, multi-model sweeps are designed to be run on **Kaggle** (using Kaggle Kernels/Notebooks).

## Step-by-Step Instructions

### 1. Create a Zip Archive of the Repository
To upload your codebase to Kaggle, zip the directory contents (excluding `.git`, `results`, and `venv` to save space):
```bash
zip -r Neural-Networks-Raw.zip . -x "venv/*" "results/*" ".git/*" ".pytest_cache/*" "__pycache__/*" "*/__pycache__/*"
```

### 2. Upload to Kaggle
1. Go to [Kaggle](https://www.kaggle.com).
2. Click **Create** -> **New Notebook**.
3. Under **File** -> **Upload data**, select the zipped file `Neural-Networks-Raw.zip`.
4. Kaggle will unzip this dataset into `/kaggle/input/neural-networks-raw/`.

### 3. Setup Environment & Install Dependencies in Notebook
In the first notebook cell, copy the codebase to a writeable directory, install dependencies, and register the package:
```python
# 1. Copy repository to writeable output space
import shutil
import os

shutil.copytree('/kaggle/input/neural-networks-raw', '/kaggle/working/Neural-Networks-Raw')
os.chdir('/kaggle/working/Neural-Networks-Raw')

# 2. Install dependencies and install package in editable mode
!pip install torch pyyaml matplotlib
!pip install -e .
```

### 4. Execute Benchmark Sweep Command
Run the multi-seed sweep using the Kaggle sweep configuration:
```python
!python experiments/phase1_associative/run_sweep.py \
  --config configs/kaggle_phase1.yaml \
  --conditions orthogonal overlapping clustered capacity_stress \
  --models mlp dense_hebbian agnis_dense agnis_kwta agnis_memory agnis_replay agnis_full_fixed \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

#### Running a Smaller Debug Sweep on Kaggle First (Optional)
If you want a fast validation check on Kaggle first:
```python
!python experiments/phase1_associative/run_sweep.py \
  --config configs/kaggle_phase1.yaml \
  --conditions overlapping \
  --models agnis_kwta agnis_full_fixed \
  --seeds 0 1
```

### 5. Aggregate Results
After the sweep finishes, run the summarizer script to generate the combined CSV and Markdown report:
```python
!python experiments/phase1_associative/summarize_phase1.py
```

### 6. Download Results
Zipping and downloading the complete result folder:
```python
!zip -r results_phase1.zip results/
```
In Kaggle Notebooks, you can download the generated `results_phase1.zip` file directly from the output files section on the right-hand panel.
