
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


REQUIRED_COLUMNS = [
    "company",
    "nse_symbol",
    "sector",
    "financial_year",
    "fiscal_year_end",
    "publication_date",
    "source_url",
    "local_pdf_path",
]


def validate_filings_manifest(
    manifest_path: str | Path,
    project_root: str | Path | None = None,
    check_files: bool = True,
    allow_empty: bool = False,
) -> dict:
    """
    Validate a CSV registry of annual-report filings.

    Dates must use YYYY-MM-DD.

    Returns:
        {
            "valid": bool,
            "rows": int,
            "errors": list[str],
        }

    The publication date should represent when the report became
    publicly available, not merely its financial year-end date.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return {
            "valid": False,
            "rows": 0,
            "errors": [
                f"Manifest file does not exist: {manifest_path}"
            ],
        }

    if project_root is None:
        # Expected layout: project_root/data/filings_manifest.csv
        root = manifest_path.resolve().parent.parent
    else:
        root = Path(project_root).resolve()

    errors = []

    try:
        df = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as error:
        return {
            "valid": False,
            "rows": 0,
            "errors": [f"Could not read manifest CSV: {error}"],
        }

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "valid": False,
            "rows": len(df),
            "errors": [
                "Missing required columns: "
                + ", ".join(missing_columns)
            ],
        }

    if df.empty:
        if not allow_empty:
            errors.append(
                "Manifest contains no filing records yet."
            )

        return {
            "valid": not errors,
            "rows": 0,
            "errors": errors,
        }

    # Normalize whitespace in registry values.
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    # Validate required values.
    for index, row in df.iterrows():
        row_number = index + 2  # Include CSV header row.

        for column in REQUIRED_COLUMNS:
            if not row[column]:
                errors.append(
                    f"Row {row_number}: '{column}' is empty."
                )

        if not all(row[column] for column in REQUIRED_COLUMNS):
            continue

        # Validate date formats and chronological order.
        try:
            year_end = datetime.strptime(
                row["fiscal_year_end"],
                "%Y-%m-%d",
            ).date()

            publication_date = datetime.strptime(
                row["publication_date"],
                "%Y-%m-%d",
            ).date()

            if publication_date <= year_end:
                errors.append(
                    f"Row {row_number}: publication_date must be "
                    "after fiscal_year_end."
                )

        except ValueError:
            errors.append(
                f"Row {row_number}: dates must use YYYY-MM-DD."
            )

        # Validate source URL syntax.
        parsed_url = urlparse(row["source_url"])

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            errors.append(
                f"Row {row_number}: source_url must be a valid "
                "HTTP or HTTPS URL."
            )

        # The manifest uses project-relative PDF paths.
        pdf_path = Path(row["local_pdf_path"])

        if pdf_path.is_absolute():
            errors.append(
                f"Row {row_number}: local_pdf_path must be "
                "relative to the project directory."
            )
            continue

        if pdf_path.suffix.lower() != ".pdf":
            errors.append(
                f"Row {row_number}: local_pdf_path must end in .pdf."
            )
            continue

        if check_files:
            resolved_pdf = (root / pdf_path).resolve()

            # Prevent paths from escaping the project directory.
            try:
                resolved_pdf.relative_to(root)
            except ValueError:
                errors.append(
                    f"Row {row_number}: local_pdf_path must remain "
                    "inside the project directory."
                )
                continue

            if not resolved_pdf.is_file():
                errors.append(
                    f"Row {row_number}: PDF file not found: "
                    f"{row['local_pdf_path']}"
                )

    # One annual report per company and financial year.
    duplicates = df.duplicated(
        subset=["nse_symbol", "financial_year"],
        keep=False,
    )

    for index in df.index[duplicates]:
        errors.append(
            f"Row {index + 2}: duplicate company and financial year "
            f"({df.loc[index, 'nse_symbol']}, "
            f"{df.loc[index, 'financial_year']})."
        )

    # Avoid accidentally registering the same PDF more than once.
    duplicate_paths = df.duplicated(
        subset=["local_pdf_path"],
        keep=False,
    )

    for index in df.index[duplicate_paths]:
        errors.append(
            f"Row {index + 2}: duplicate local_pdf_path "
            f"({df.loc[index, 'local_pdf_path']})."
        )

    return {
        "valid": not errors,
        "rows": len(df),
        "errors": errors,
    }
