"""Convenience wrapper around ``python -m sage`` for the paper's benchmarks.

Example
-------
    python scripts/run_benchmark.py --preset math500 \
        --input data/math500.jsonl --output out/math500_sage.jsonl \
        --model-name Qwen/Qwen3-8B-FP8

For reasoning-mode benchmarks (e.g. AIME) add ``--engine-type think``.
"""
from sage.__main__ import main

if __name__ == "__main__":
    main()
