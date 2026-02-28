"""
Demo flow for metaflow-card-dataprofile.

Run with:
    python tests/test_flow.py run

Then view the card with:
    python tests/test_flow.py card view --origin-run-id <run_id> --a start
    python tests/test_flow.py card view --origin-run-id <run_id> --a split

Exercises:
  - start    : single DataFrame, large sample warning
  - split    : two DataFrames → tabbed layout
  - compare  : comparison mode (train vs test distributions)
  - edge     : empty DataFrame + null-heavy DataFrame (graceful error handling)
  - explicit : inline DataProfileComponent usage
"""

import numpy as np
import pandas as pd
from metaflow import FlowSpec, card, current, step

from metaflow_dataprofiler import DataProfileComponent


def _make_population(n, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 75, n),
            "income": rng.exponential(52_000, n).round(2),
            "credit_score": rng.integers(300, 850, n),
            "loan_amount": rng.exponential(15_000, n).round(2),
            "employment_years": rng.integers(0, 40, n),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "has_default": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "null_col": np.where(rng.random(n) > 0.3, rng.standard_normal(n), np.nan),
        }
    )


class DataProfileDemoFlow(FlowSpec):
    """Demo flow showcasing all features of metaflow-card-dataprofile."""

    # ── single large DataFrame (triggers sample warning) ─────────────────
    @card(type="dataprofile", options={"sample": 1_000, "minimal": True})
    @step
    def start(self):
        """Single DataFrame with sampling warning."""
        self.data = _make_population(n=5_000, seed=0)
        self.next(self.split)

    # ── two DataFrames → automatic tabbed layout ──────────────────────────
    @card(type="dataprofile", options={"minimal": True})
    @step
    def split(self):
        """Train/test split — two DataFrames rendered as separate tabs."""
        full = _make_population(n=2_000, seed=1)
        split_idx = int(len(full) * 0.8)
        self.train_df = full.iloc[:split_idx].reset_index(drop=True)
        self.test_df = full.iloc[split_idx:].reset_index(drop=True)
        self.next(self.compare)

    # ── side-by-side distribution comparison ─────────────────────────────
    @card(
        type="dataprofile",
        options={
            "compare": ["train_df", "test_df"],
            "minimal": True,
        },
    )
    @step
    def compare(self):
        """Compare train vs test distributions side-by-side."""
        rng = np.random.default_rng(99)
        n_train, n_test = 800, 200

        # Introduce mild distributional skew in test set
        self.train_df = pd.DataFrame(
            {
                "age": rng.normal(35, 10, n_train).clip(18, 75).astype(int),
                "income": rng.exponential(50_000, n_train).round(2),
                "score": rng.uniform(0, 1, n_train),
            }
        )
        self.test_df = pd.DataFrame(
            {
                "age": rng.normal(45, 12, n_test).clip(18, 75).astype(int),  # older!
                "income": rng.exponential(70_000, n_test).round(2),           # richer!
                "score": rng.uniform(0, 1, n_test),
            }
        )
        self.next(self.edge)

    # ── edge cases: empty + null-heavy ────────────────────────────────────
    @card(type="dataprofile", options={"minimal": True})
    @step
    def edge(self):
        """Empty DataFrame shows placeholder; null-heavy DataFrame still profiles."""
        self.empty_df = pd.DataFrame()

        rng = np.random.default_rng(7)
        n = 200
        # 90 % nulls in every column
        mask = rng.random((n, 3)) > 0.1
        self.sparse_df = pd.DataFrame(
            np.where(mask, rng.standard_normal((n, 3)), np.nan),
            columns=["a", "b", "c"],
        )
        self.next(self.explicit)

    # ── inline DataProfileComponent ───────────────────────────────────────
    @card
    @step
    def explicit(self):
        """Explicit before/after profiling using DataProfileComponent."""
        rng = np.random.default_rng(5)
        raw = pd.DataFrame(
            {
                "value": np.concatenate(
                    [rng.normal(0, 1, 150), rng.normal(10, 1, 50)]  # bimodal
                ),
                "category": rng.choice(["A", "B", "C"], 200),
                "noise": np.where(rng.random(200) > 0.5, rng.random(200), np.nan),
            }
        )

        current.card.append(DataProfileComponent(raw, title="Before cleaning", minimal=True))

        # "Clean": drop rows where value > 5 (removes the second cluster)
        cleaned = raw[raw["value"] < 5].copy()
        current.card.append(DataProfileComponent(cleaned, title="After cleaning", minimal=True))

        self.cleaned = cleaned
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DataProfileDemoFlow()
