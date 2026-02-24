"""
error_analysis.py
-----------------
Character-level error analysis using DP edit-distance traceback.
Classifies errors as substitutions, insertions, or deletions.
"""

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def character_error_analysis(
    df: pd.DataFrame,
    ref_col: str = "ground_truth",
    hyp_col: str = "prediction",
) -> tuple[dict, dict, dict]:
    """
    Analyse character-level errors across all rows in *df*.

    Parameters
    ----------
    df      : DataFrame with at least ``ref_col`` and ``hyp_col`` columns.
    ref_col : Column name for ground-truth strings.
    hyp_col : Column name for predicted strings.

    Returns
    -------
    (substitutions, insertions, deletions) — each a defaultdict(int).
      substitutions : {(ref_char, hyp_char): count}
      insertions    : {hyp_char: count}
      deletions     : {ref_char: count}
    """
    substitutions: dict = defaultdict(int)
    insertions:    dict = defaultdict(int)
    deletions:     dict = defaultdict(int)

    for _, row in df.iterrows():
        ref = row[ref_col].lower()
        hyp = row[hyp_col].lower()
        _traceback_errors(ref, hyp, substitutions, insertions, deletions)

    return substitutions, insertions, deletions


def _traceback_errors(ref, hyp, substitutions, insertions, deletions):
    """Fill edit-distance DP table and traceback to classify every edit."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions[(ref[i - 1], hyp[j - 1])] += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions[hyp[j - 1]] += 1
            j -= 1
        else:
            deletions[ref[i - 1]] += 1
            i -= 1


def plot_error_analysis(substitutions: dict, insertions: dict, deletions: dict, top_n: int = 8):
    """Bar charts for the top-N substitution, deletion, and insertion errors."""
    top_subs = sorted(substitutions.items(), key=lambda x: -x[1])[:top_n]
    top_dels = sorted(deletions.items(),     key=lambda x: -x[1])[:top_n]
    top_ins  = sorted(insertions.items(),    key=lambda x: -x[1])[:top_n]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("🔍 Character-level Error Analysis", fontsize=15, fontweight="bold")

    _hbar(axes[0], [f"'{a}'→'{b}'" for (a, b), _ in top_subs],
          [c for _, c in top_subs], "#EF4444", "Substitution Errors")
    _hbar(axes[1], [f"'{c}'" for c, _ in top_dels],
          [cnt for _, cnt in top_dels], "#F97316", "Deletion Errors (missed)")
    _hbar(axes[2], [f"'{c}'" for c, _ in top_ins],
          [cnt for _, cnt in top_ins], "#8B5CF6", "Insertion Errors (extra)")

    plt.tight_layout()
    plt.show()


def _hbar(ax, labels, counts, color, title):
    if not labels:
        ax.text(0.5, 0.5, "No errors!", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.barh(labels[::-1], counts[::-1], color=color, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title(title, fontweight="bold")
