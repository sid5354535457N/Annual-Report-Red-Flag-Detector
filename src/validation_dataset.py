
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

ALLOWED_LABELS = {
    "true_positive",
    "false_positive",
    "needs_review",
    "disclosure_only",
}

ALLOWED_REVIEW_STATUSES = {
    "pending_manual_review",
    "reviewed",
    "requires_revision",
}

DUPLICATE_KEY_COLUMNS = [
    "company",
    "financial_year",
    "source_pdf",
    "pdf_page",
    "category_key",
]


def _empty_result(errors=None):
    """Return a consistent result for an unreadable dataset."""

    return {
        "valid": False,
        "rows": 0,
        "columns": [],
        "errors": list(errors or []),
        "label_counts": {},
        "review_status_counts": {},
        "pending_manual_review": 0,
    }


def _resolve_source_pdf(source_pdf, project_root):
    """Resolve a source report path inside or outside data/reports."""

    source_path = Path(str(source_pdf).strip())

    if source_path.is_absolute():
        return source_path

    direct_path = project_root / source_path

    if direct_path.is_file():
        return direct_path

    return project_root / "data" / "reports" / source_path


def validate_validation_dataset(
    csv_path="data/validation_labels.csv",
    project_root=None,
    check_source_files=True,
):
    """
    Validate the structure and basic integrity of the human-label dataset.

    This checks data quality, not whether the suggested labels are
    factually correct. Manual review is still necessary.
    """

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )

    path = Path(csv_path)

    if not path.is_absolute():
        path = root / path

    errors = []

    if not path.is_file():
        return _empty_result(
            [f"Dataset file not found: {path}"]
        )

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return _empty_result(
            [f"Could not read dataset: {exc}"]
        )

    columns = df.columns.tolist()

    if df.empty:
        errors.append("Dataset contains no data rows.")

    if len(columns) == 1:
        errors.append(
            "Dataset has only one column. Check the CSV delimiter "
            "and quoting."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Do not perform column-based checks until the schema is present.
    if missing_columns:
        return {
            "valid": False,
            "rows": len(df),
            "columns": columns,
            "errors": errors,
            "label_counts": {},
            "review_status_counts": {},
            "pending_manual_review": 0,
        }

    # Check that required cells have values.
    for column in REQUIRED_COLUMNS:
        blank_mask = (
            df[column].isna()
            | df[column].astype(str).str.strip().eq("")
        )

        for index in df.index[blank_mask]:
            errors.append(
                f"Row {index + 2}: '{column}' is blank."
            )

    # Validate labels without silently changing them.
    labels = df["suggested_label"].fillna("").astype(str).str.strip()

    for index, label in labels.items():
        if label not in ALLOWED_LABELS:
            errors.append(
                f"Row {index + 2}: invalid suggested_label "
                f"'{label}'."
            )

    statuses = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for index, status in statuses.items():
        if status not in ALLOWED_REVIEW_STATUSES:
            errors.append(
                f"Row {index + 2}: invalid review_status "
                f"'{status}'."
            )

    # Page numbers must be positive integers.
    for column in ["pdf_page", "printed_page"]:
        numeric = pd.to_numeric(df[column], errors="coerce")

        invalid_mask = (
            numeric.isna()
            | (numeric <= 0)
            | ((numeric % 1) != 0)
        )

        for index in df.index[invalid_mask]:
            errors.append(
                f"Row {index + 2}: '{column}' must be "
                "a positive integer."
            )

    # Each category may occur on a page, but identical records
    # should not be duplicated in the validation dataset.
    duplicate_mask = df.duplicated(
        subset=DUPLICATE_KEY_COLUMNS,
        keep=False,
    )

    for index in df.index[duplicate_mask]:
        errors.append(
            f"Row {index + 2}: duplicate record key "
            f"{DUPLICATE_KEY_COLUMNS}."
        )

    # Verify that source reports are actually available locally.
    if check_source_files:
        for index, source_pdf in df["source_pdf"].items():
            if pd.isna(source_pdf) or not str(source_pdf).strip():
                continue

            resolved_pdf = _resolve_source_pdf(
                source_pdf,
                root,
            )

            if not resolved_pdf.is_file():
                errors.append(
                    f"Row {index + 2}: source PDF not found: "
                    f"{source_pdf}"
                )

    label_counts = labels.value_counts().to_dict()
    status_counts = statuses.value_counts().to_dict()

    pending_count = int(
        (statuses == "pending_manual_review").sum()
    )

    return {
        "valid": len(errors) == 0,
        "rows": len(df),
        "columns": columns,
        "errors": errors,
        "label_counts": label_counts,
        "review_status_counts": status_counts,
        "pending_manual_review": pending_count,
    }
