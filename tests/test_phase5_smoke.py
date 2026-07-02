"""
Phase 5 Smoke Test run verification.
"""

import subprocess
import os

def test_benchmark_smoke_runs():
    """Verify that the benchmark script runs under smoke conditions."""
    script_path = "experiments/phase5_tinystories/run_tinystories_benchmark.py"
    
    cmd = [
        "python",
        script_path,
        "--model", "seq_agnis_neurogenesis",
        "--seed", "0",
        "--config", "configs/phase5_smoke.yaml",
        "--smoke"
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Benchmark run failed:\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    assert "Run finished successfully" in res.stdout
