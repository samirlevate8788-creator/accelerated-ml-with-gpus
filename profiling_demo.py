"""
cuDF Profiling Demonstration
Google Cloud & NVIDIA Codelab Implementation

Highlights:
1. High-level profiling (%%cudf.pandas.profile)
2. Line-by-line GPU/CPU fallback profiling (%%cudf.pandas.line_profile)
3. Optimizing slow row-level python .apply() fallbacks into 100% GPU vectorized code
"""

import glob
import os
import time
import pandas as pd

def demonstrate_vectorization_speedup(data_dir="nyc_taxi_data"):
    """Demonstrates how replacing row-by-row apply with vectorized pd.cut speeds up execution."""
    files = glob.glob(os.path.join(data_dir, "*-01.parquet"))
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        print("[INFO] No parquet data found to run live demo.")
        return

    print("[PROFILE] Loading sample for profiling demonstration...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sample(10_000, random_state=42)
    df['hour'] = df['tpep_pickup_datetime'].dt.hour

    # Approach 1: Slow custom python apply (falls back to CPU in cudf.pandas)
    def categorize_hour(hour):
        if hour < 12:
            return 'Morning'
        else:
            return 'Afternoon/Evening'

    t0 = time.perf_counter()
    df['time_of_day_slow'] = df['hour'].apply(categorize_hour)
    t_slow = time.perf_counter() - t0
    print(f"Row-level .apply() (CPU fallback):      {t_slow:.4f}s")

    # Approach 2: Fast vectorized pd.cut (stays 100% on GPU in cuDF)
    t0 = time.perf_counter()
    cut_bins = [-1, 11, 24]
    cut_labels = ['Morning', 'Afternoon/Evening']
    df['time_of_day_fast'] = pd.cut(df['hour'], bins=cut_bins, labels=cut_labels)
    t_fast = time.perf_counter() - t0
    print(f"Vectorized pd.cut() (Native GPU execution): {t_fast:.4f}s")

    speedup = t_slow / max(t_fast, 0.00001)
    print(f"Speedup achieved by avoiding CPU fallback: {speedup:.1f}x faster!")

if __name__ == "__main__":
    print("=" * 60)
    print(" cuDF Profiling & CPU-Fallback Optimization Demo")
    print("=" * 60)
    print("\nIn Jupyter / Colab notebooks, run:")
    print("  %load_ext cudf.pandas")
    print("  %%cudf.pandas.profile       <-- High-level overview of GPU vs CPU calls")
    print("  %%cudf.pandas.line_profile  <-- Line-by-line GPU vs CPU execution counts")
    print("=" * 60 + "\n")
    demonstrate_vectorization_speedup()
