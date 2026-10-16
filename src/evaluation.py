
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "company",
    "financial_year",
    "source_pdf",
    "pdf_page",
    "printed_page",
    "category_key",
    "detector_type",
    "reported_amount_or_status",
    "event_id",
    "suggested_label",
    "review_status",
    "review_notes",
]

BINARY_LABELS = {
    "true_positive",
    "false_positive",
}

MIN_CASES_FOR_PRECISION = 10


def evaluate_validation_dataset(
    csv_path="data/validation_labels.csv",
    min_cases=MIN_CASES_FOR_PRECISION,
):
    """
    Summarize manually reviewed validation records.

    Precision is reported only when enough binary-labelled risk-signal
    records are available. This minimum is a project safeguard, not
    a universal statistical standard.

    Recall cannot be measured from detected findings alone because
    independently discovered events missed by the detector are absent.
    """

    path = Path(csv_path)
    errors = []

    if not path.is_file():
        return {
            "valid": False,
            "errors": [f"Dataset file not found: {path}"],
        }

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return {
            "valid": False,
            "errors": [f"Could not read dataset: {exc}"],
        }

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "valid": False,
            "errors": [
                "Missing required columns: "
                + ", ".join(missing_columns)
            ],
        }

    if min_cases < 1:
        return {
            "valid": False,
            "errors": ["min_cases must be at least 1."],
        }

    # Keep only records marked as reviewed by the user.
    reviewed = df[
        df["review_status"].astype(str).str.strip().eq("reviewed")
    ].copy()

    pending = df[
        df["review_status"]
        .astype(str)
        .str.strip()
        .eq("pending_manual_review")
    ]

    # Precision for scored signals must be evaluated separately from
    # ordinary disclosures and review-only items.
    risk_signals = reviewed[
        reviewed["detector_type"]
        .astype(str)
        .str.strip()
        .eq("risk_signal")
    ].copy()

    # Only reviewed binary outcomes are usable for this calculation.
    binary_cases = risk_signals[
        risk_signals["suggested_label"]
        .astype(str)
        .str.strip()
        .isin(BINARY_LABELS)
    ].copy()

    labels = (
        reviewed["suggested_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    true_positives = int(
        binary_cases["suggested_label"]
        .astype(str)
        .str.strip()
        .eq("true_positive")
        .sum()
    )

    false_positives = int(
        binary_cases["suggested_label"]
        .astype(str)
        .str.strip()
        .eq("false_positive")
        .sum()
    )

    binary_count = len(binary_cases)

    # Withhold the metric for very small samples.
    precision = None

    if binary_count >= min_cases:
        precision = (
            true_positives / binary_count
            if binary_count
            else None
        )
        precision_note = (
            "Descriptive precision on reviewed risk signals only. "
            "It may not generalize to other companies or years."
        )
    else:
        precision_note = (
            f"Not reported: {binary_count} binary-labelled risk signals; "
            f"at least {min_cases} are required by this project's "
            "current reporting threshold."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_records": len(df),
        "reviewed_records": len(reviewed),
        "pending_records": len(pending),
        "reviewed_label_counts": labels.value_counts().to_dict(),
        "reviewed_risk_signal_records": len(risk_signals),
        "binary_labelled_risk_signals": binary_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": precision,
        "precision_note": precision_note,
        "recall": None,
        "recall_note": (
            "Unavailable: this dataset contains detector findings, "
            "not an independent inventory of all true risk events. "
            "Review report sections where the detector found nothing "
            "to identify possible false negatives."
        ),
    }


def main():
    """Run the evaluator and print a readable report."""

    result = evaluate_validation_dataset()

    print("\n========== VALIDATION EVALUATION ==========")

    if not result["valid"]:
        print("Dataset validation failed.")

        for error in result["errors"]:
            print("-", error)

        raise SystemExit(1)

    print("Dataset records:", result["total_records"])
    print("Reviewed records:", result["reviewed_records"])
    print("Pending records:", result["pending_records"])

    print("\nReviewed label counts:")

    for label, count in sorted(
        result["reviewed_label_counts"].items()
    ):
        print(f"  {label}: {count}")

    print("\nRisk-signal evaluation:")
    print(
        "Reviewed risk-signal records:",
        result["reviewed_risk_signal_records"],
    )
    print(
        "Binary-labelled risk signals:",
        result["binary_labelled_risk_signals"],
    )
    print("True positives:", result["true_positives"])
    print("False positives:", result["false_positives"])

    if result["precision"] is None:
        print("Precision: NOT REPORTED")
    else:
        print(f"Precision: {result['precision']:.1%}")

    print("Precision note:", result["precision_note"])
    print("\nRecall: NOT AVAILABLE")
    print("Recall note:", result["recall_note"])


if __name__ == "__main__":
    main()
