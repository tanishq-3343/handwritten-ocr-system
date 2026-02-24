"""
visualizations.py
-----------------
All plotting utilities for the Handwritten OCR System.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PALETTE = {"neat": "#3B82F6", "cursive": "#8B5CF6", "mixed": "#10B981"}
STYLES  = ["neat", "cursive", "mixed"]


# ---------------------------------------------------------------------------
# Sample image grid
# ---------------------------------------------------------------------------

def plot_sample_images(dataset: list[dict], per_style: int = 2) -> None:
    """Show a grid of sample handwriting images grouped by style."""
    fig, axes = plt.subplots(3, per_style, figsize=(16, 9))
    fig.suptitle("📝 Sample Handwriting Images by Style", fontsize=18, fontweight="bold", y=1.02)
    for row_idx, style in enumerate(STYLES):
        samples = [d for d in dataset if d["style"] == style][:per_style]
        for col_idx, sample in enumerate(samples):
            ax = axes[row_idx][col_idx]
            ax.imshow(sample["image"])
            label = sample["ground_truth"][:45] + ("..." if len(sample["ground_truth"]) > 45 else "")
            ax.set_title(f"[{style.upper()}]  {label}", fontsize=9,
                         color=PALETTE[style], fontweight="bold")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Performance dashboard (9 sub-plots)
# ---------------------------------------------------------------------------

def plot_dashboard(df: pd.DataFrame, style_summary: pd.DataFrame) -> None:
    """Full 3×3 performance dashboard."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("📊 Handwritten OCR — Comprehensive Performance Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1 – CER & WER bar chart
    ax1 = fig.add_subplot(3, 3, 1)
    _grouped_bar(ax1, style_summary, "Mean_CER", "Mean_WER",
                 "#EF4444", "#F97316", "CER", "WER", "CER & WER by Style")

    # 2 – Char & Word Accuracy
    ax2 = fig.add_subplot(3, 3, 2)
    _grouped_bar(ax2, style_summary, "Char_Accuracy", "Word_Accuracy",
                 "#22C55E", "#3B82F6", "Char Acc", "Word Acc",
                 "Char & Word Accuracy by Style")

    # 3 – Box plots
    ax3 = fig.add_subplot(3, 3, 3)
    data_for_box = [df[df["style"] == s]["cer"].values for s in STYLES]
    bp = ax3.boxplot(data_for_box, labels=[s.capitalize() for s in STYLES],
                     patch_artist=True)
    for patch, style in zip(bp["boxes"], STYLES):
        patch.set_facecolor(PALETTE[style]); patch.set_alpha(0.7)
    ax3.set_ylabel("CER"); ax3.set_title("CER Distribution", fontweight="bold")

    # 4 – Scatter CER vs text length
    ax4 = fig.add_subplot(3, 3, 4)
    for style in STYLES:
        sub = df[df["style"] == style]
        ax4.scatter(sub["ground_truth"].str.len(), sub["cer"],
                    label=style.capitalize(), color=PALETTE[style],
                    alpha=0.75, s=60, edgecolors="white", linewidth=0.5)
    ax4.set_xlabel("Text Length (chars)"); ax4.set_ylabel("CER")
    ax4.set_title("CER vs Text Length", fontweight="bold"); ax4.legend()

    # 5 – Radar chart
    ax5 = fig.add_subplot(3, 3, 5, polar=True)
    _radar(ax5, style_summary)

    # 6 – Overall accuracy donut
    ax6 = fig.add_subplot(3, 3, 6)
    overall_acc = df["char_accuracy"].mean()
    ax6.pie([overall_acc, 1 - overall_acc], labels=["Correct", "Error"],
            colors=["#22C55E", "#EF4444"], autopct="%1.1f%%",
            startangle=90, wedgeprops={"width": 0.5})
    ax6.set_title("Overall Character Accuracy", fontweight="bold")

    # 7 – Engine usage
    ax7 = fig.add_subplot(3, 3, 7)
    mc = df["method"].value_counts()
    ax7.pie(mc.values, labels=mc.index, colors=["#6366F1", "#EC4899"],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax7.set_title("OCR Engine Usage (Ensemble)", fontweight="bold")

    # 8 – Heatmap
    ax8 = fig.add_subplot(3, 3, 8)
    heat_data = style_summary[["Mean_CER", "Mean_WER", "Char_Accuracy", "Word_Accuracy"]].rename(
        columns={"Mean_CER": "CER", "Mean_WER": "WER",
                 "Char_Accuracy": "Char Acc", "Word_Accuracy": "Word Acc"})
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=ax8, linewidths=0.5, vmin=0, vmax=1, annot_kws={"fontsize": 10})
    ax8.set_title("Metric Heatmap by Style", fontweight="bold")
    ax8.set_yticklabels([s.capitalize() for s in heat_data.index], rotation=0)

    # 9 – Per-sample accuracy
    ax9 = fig.add_subplot(3, 3, 9)
    colors_list = [PALETTE[s] for s in df["style"]]
    ax9.bar(df["id"], df["char_accuracy"], color=colors_list, alpha=0.8, edgecolor="white")
    mean_acc = df["char_accuracy"].mean()
    ax9.axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean={mean_acc:.2f}")
    ax9.set_xlabel("Sample ID"); ax9.set_ylabel("Char Accuracy")
    ax9.set_title("Per-sample Character Accuracy", fontweight="bold")
    patches = [mpatches.Patch(color=PALETTE[s], label=s.capitalize()) for s in STYLES]
    ax9.legend(handles=patches + [plt.Line2D([0], [0], color="red",
               linestyle="--", label="Mean")], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ---------------------------------------------------------------------------
# Engine comparison
# ---------------------------------------------------------------------------

def plot_engine_comparison(cmp_df: pd.DataFrame) -> None:
    """Side-by-side Tesseract vs EasyOCR bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("🆚 Tesseract vs EasyOCR by Handwriting Style",
                 fontsize=14, fontweight="bold")
    for ax, (tess_m, easy_m), title in zip(
        axes,
        [("tesseract_char_acc", "easyocr_char_acc"), ("tesseract_cer", "easyocr_cer")],
        ["Character Accuracy", "Character Error Rate"],
    ):
        grp = cmp_df.groupby("style")[[tess_m, easy_m]].mean()
        x = np.arange(len(grp))
        ax.bar(x - 0.2, grp[tess_m], 0.35, label="Tesseract",
               color="#6366F1", alpha=0.85, edgecolor="white")
        ax.bar(x + 0.2, grp[easy_m], 0.35, label="EasyOCR",
               color="#EC4899", alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in grp.index])
        ax.set_title(title, fontweight="bold"); ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Qualitative results
# ---------------------------------------------------------------------------

def show_ocr_results(df_results: pd.DataFrame, dataset: list[dict], num_samples: int = 6) -> None:
    """Display images with ground-truth and predicted text overlaid."""
    rows = df_results.sample(min(num_samples, len(df_results)), random_state=42)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(16, n * 2.8))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        ax.imshow(dataset[row["id"]]["image"])
        ax.axis("off")
        color = PALETTE[row["style"]]
        ax.set_title(
            f"[{row['style'].upper()}]  Char Acc: {row['char_accuracy']:.2f} | "
            f"Word Acc: {row['word_accuracy']:.2f} | CER: {row['cer']:.3f} | "
            f"Engine: {row['method']}",
            fontsize=9, color=color, fontweight="bold", pad=3,
        )
        ax.text(0.01, -0.04, f"GT : {row['ground_truth']}",
                transform=ax.transAxes, fontsize=8, color="#166534", va="top")
        pred_color = "#991B1B" if row["cer"] > 0.3 else "#1D4ED8"
        ax.text(0.01, -0.12, f"OCR: {row['prediction']}",
                transform=ax.transAxes, fontsize=8, color=pred_color, va="top")
    plt.suptitle("🖼️ Qualitative OCR Results — Ground Truth vs Prediction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grouped_bar(ax, style_summary, col_a, col_b, color_a, color_b, label_a, label_b, title):
    x = np.arange(len(STYLES)); w = 0.35
    vals_a = [style_summary.loc[s, col_a] if s in style_summary.index else 0 for s in STYLES]
    vals_b = [style_summary.loc[s, col_b] if s in style_summary.index else 0 for s in STYLES]
    ax.bar(x - w / 2, vals_a, w, label=label_a, color=color_a, alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, vals_b, w, label=label_b, color=color_b, alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in STYLES])
    ax.legend(); ax.set_ylim(0, 1); ax.set_title(title, fontweight="bold")


def _radar(ax, style_summary):
    categories = ["Char Acc", "Word Acc", "1-CER", "1-WER", "Exact\nMatch %"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    for style in STYLES:
        if style not in style_summary.index:
            continue
        row = style_summary.loc[style]
        em_pct = row["Exact_Matches"] / row["Samples"]
        values = [row["Char_Accuracy"], row["Word_Accuracy"],
                  1 - row["Mean_CER"], 1 - row["Mean_WER"], em_pct] + [row["Char_Accuracy"]]
        ax.plot(angles, values, "o-", linewidth=2, color=PALETTE[style], label=style.capitalize())
        ax.fill(angles, values, alpha=0.15, color=PALETTE[style])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Radar: Multi-metric by Style", fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
