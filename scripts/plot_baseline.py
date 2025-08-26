from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from metrics_backend import parse_metrics


def load_summaries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics and return summaries split by prevalence."""
    rows = parse_metrics(limit=0, since_day=None)
    if not rows:
        raise SystemExit("No metrics available. Set METRICS_PATH to your metrics.jsonl file.")

    df = pd.DataFrame(rows)
    df["lift_abs"] = df["ap_micro"] - df["prev_micro"]

    # Split into high/low prevalence groups using the median prevalence as threshold
    median_prev = df["prev_micro"].median()
    df["group"] = df["prev_micro"].apply(lambda x: "high" if x >= median_prev else "low")

    summary_high = df[df["group"] == "high"].groupby("epoch")[
        ["ap_micro", "prev_micro", "lift_abs"]
    ].mean()
    summary_low = df[df["group"] == "low"].groupby("epoch")[
        ["ap_micro", "prev_micro", "lift_abs"]
    ].mean()

    return summary_high, summary_low


def plot(summary_high: pd.DataFrame, summary_low: pd.DataFrame) -> None:
    """Create APμ vs prevalence and lift plots."""
    plt.figure(figsize=(10, 5))
    plt.plot(summary_high.index, summary_high["ap_micro"], label="High Prev - APμ", marker="o")
    plt.plot(summary_low.index, summary_low["ap_micro"], label="Low Prev - APμ", marker="o")

    plt.plot(
        summary_high.index,
        summary_high["prev_micro"],
        label="High Prev Baseline",
        linestyle="--",
    )
    plt.plot(
        summary_low.index,
        summary_low["prev_micro"],
        label="Low Prev Baseline",
        linestyle="--",
    )

    plt.title("AP Micro vs Prevalence Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 5))
    plt.plot(summary_high.index, summary_high["lift_abs"], label="High Prev - Lift", marker="o")
    plt.plot(summary_low.index, summary_low["lift_abs"], label="Low Prev - Lift", marker="o")

    plt.title("Lift over Baseline by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Lift (APμ - prevμ)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    summary_high, summary_low = load_summaries()
    plot(summary_high, summary_low)


if __name__ == "__main__":
    main()
