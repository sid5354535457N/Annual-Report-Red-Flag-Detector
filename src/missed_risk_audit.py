
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.pdf_parser import extract_pdf_pages
from src.rules import (
    RULES,
    detect_red_flags,
    is_excluded_match,
    is_negated,
)


AUDIT_COLUMNS = [
    "company",
    "financial_year",
    "source_pdf",
    "pdf_page",
    "printed_page",
    "section_group",
    "detector_findings",
    "detector_match_status",
    "detector_match_categories",
    "category_limit_suppressed",
    "page_preview",
    "manual_review_status",
    "manual_risk_found",
    "manual_event_category",
    "event_description",
    "detector_missed_event",
    "review_notes",
]

DEFAULT_SECTION_RANGES = [
    (169, 238, "consolidated_financial_statements"),
    (250, 309, "standalone_financial_statements"),
]

ALLOWED_REVIEW_STATUSES = {
    "pending_manual_review",
    "reviewed",
    "requires_revision",
}

ALLOWED_MANUAL_RISK_VALUES = {
    "",
    "yes",
    "no",
    "uncertain",
}

ALLOWED_MISSED_EVENT_VALUES = {
    "",
    "yes",
    "no",
    "uncertain",
    "not_applicable",
}

ALLOWED_MATCH_STATUSES = {
    "no_rule_match",
    "suppressed_by_category_limit",
    "matched_but_not_reported",
}


def normalize_text(text: str) -> str:
    """Normalize whitespace extracted from PDF pages."""
    return re.sub(r"\s+", " ", text).strip()


def find_rule_matches_by_page(pages: list[dict]) -> dict[int, set[str]]:
    """
    Identify configured rule matches without the three-findings-per-
    category output cap. Existing exclusions and negation checks apply.
    """

    matches_by_page = {}

    for page_data in pages:
        try:
            page_number = int(page_data.get("page"))
        except (TypeError, ValueError):
            continue

        text = page_data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        matched_categories = set()

        for rule in RULES:
            for pattern in rule["patterns"]:
                for match in re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    if is_excluded_match(
                        rule["key"],
                        text,
                        match.start(),
                        match.end(),
                    ):
                        continue

                    if is_negated(
                        rule["key"],
                        text,
                        match.start(),
                        match.end(),
                    ):
                        continue

                    matched_categories.add(rule["key"])
                    break

                if rule["key"] in matched_categories:
                    break

        matches_by_page[page_number] = matched_categories

    return matches_by_page


def classify_match_status(
    matched_categories: set[str],
    reported_categories: set[str],
    reported_category_counts: Counter,
) -> tuple[str, str]:
    """Distinguish genuine no-match pages from unreported matches."""

    if not matched_categories:
        return "no_rule_match", ""

    unreported_categories = matched_categories - reported_categories

    if not unreported_categories:
        return "no_rule_match", ""

    suppressed_categories = sorted(
        category
        for category in unreported_categories
        if reported_category_counts[category] >= 3
    )

    if suppressed_categories:
        return (
            "suppressed_by_category_limit",
            ";".join(suppressed_categories),
        )

    return "matched_but_not_reported", ""


def sample_evenly_by_section(
    rows: list[dict],
    sample_size: int,
    rng: random.Random,
) -> list[dict]:
    """Select a reproducible sample spread across section groups."""

    if sample_size <= 0 or not rows:
        return []

    grouped = {}

    for row in rows:
        grouped.setdefault(row["section_group"], []).append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: row["pdf_page"])

    selected = []
    groups = list(grouped)

    while len(selected) < sample_size:
        progressed = False

        for group in groups:
            if not grouped[group]:
                continue

            selected.append(rng.choice(grouped[group]))
            grouped[group].remove(selected[-1])
            progressed = True

            if len(selected) >= sample_size:
                break

        if not progressed:
            break

    return selected


def build_missed_risk_audit_from_pages(
    pages,
    output_csv="data/missed_risk_audit.csv",
    company="TCS",
    financial_year="2024-25",
    source_pdf="TCS_FY2024-25.pdf",
    sample_size=12,
    random_seed=42,
    section_ranges=None,
):
    """
    Create an audit sample of pages without displayed detector findings.

    Raw matches and category-limit suppression are retained separately.
    No manual labels are filled automatically.
    """

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    if section_ranges is None:
        section_ranges = DEFAULT_SECTION_RANGES

    result = detect_red_flags(pages)

    pages_with_reported_findings = set()
    reported_categories_by_page = {}
    reported_category_counts = Counter()

    for finding in result["findings"]:
        try:
            page_number = int(finding["page"])
        except (TypeError, ValueError):
            continue

        category = finding["category_key"]
        pages_with_reported_findings.add(page_number)
        reported_categories_by_page.setdefault(
            page_number, set()
        ).add(category)
        reported_category_counts[category] += 1

    raw_matches_by_page = find_rule_matches_by_page(pages)
    candidates = {}

    for page_data in pages:
        try:
            page_number = int(page_data.get("page"))
        except (TypeError, ValueError):
            continue

        if page_number in pages_with_reported_findings:
            continue

        text = page_data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        section_group = None
        for start_page, end_page, group_name in section_ranges:
            if start_page <= page_number <= end_page:
                section_group = group_name
                break

        if section_group is None:
            continue

        raw_categories = raw_matches_by_page.get(
            page_number, set()
        )
        reported_categories = reported_categories_by_page.get(
            page_number, set()
        )

        match_status, suppressed_categories = classify_match_status(
            raw_categories,
            reported_categories,
            reported_category_counts,
        )

        normalized = normalize_text(text)
        preview = normalized[:600]
        if len(normalized) > 600:
            preview += "..."

        candidates[page_number] = {
            "company": company,
            "financial_year": financial_year,
            "source_pdf": Path(source_pdf).name,
            "pdf_page": page_number,
            "printed_page": "",
            "section_group": section_group,
            "detector_findings": "none",
            "detector_match_status": match_status,
            "detector_match_categories": ";".join(
                sorted(raw_categories)
            ),
            "category_limit_suppressed": suppressed_categories,
            "page_preview": preview,
            "manual_review_status": "pending_manual_review",
            "manual_risk_found": "",
            "manual_event_category": "",
            "event_description": "",
            "detector_missed_event": "",
            "review_notes": "",
        }

    if not candidates:
        raise ValueError(
            "No eligible pages without displayed findings were found "
            "in the configured section ranges."
        )

    rng = random.Random(random_seed)
    target_size = min(sample_size, len(candidates))

    priority_candidates = [
        row for row in candidates.values()
        if row["detector_match_status"] != "no_rule_match"
    ]
    no_match_candidates = [
        row for row in candidates.values()
        if row["detector_match_status"] == "no_rule_match"
    ]

    # Include suppressed matches first, then supplement with true
    # no-rule-match pages if more sample slots remain.
    if len(priority_candidates) > target_size:
        selected = rng.sample(priority_candidates, target_size)
    else:
        selected = list(priority_candidates)
        selected.extend(
            sample_evenly_by_section(
                no_match_candidates,
                target_size - len(selected),
                rng,
            )
        )

    group_order = {
        group_name: index
        for index, (_, _, group_name) in enumerate(section_ranges)
    }

    selected.sort(
        key=lambda row: (
            group_order.get(row["section_group"], 999),
            row["pdf_page"],
        )
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(selected, columns=AUDIT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return {
        "output_csv": str(output_path),
        "total_pdf_pages": len(pages),
        "pages_with_reported_findings": len(
            pages_with_reported_findings
        ),
        "eligible_pages_without_reported_findings": len(candidates),
        "no_rule_match_pages_available": sum(
            row["detector_match_status"] == "no_rule_match"
            for row in candidates.values()
        ),
        "suppressed_pages_available": sum(
            row["detector_match_status"]
            == "suppressed_by_category_limit"
            for row in candidates.values()
        ),
        "rows_written": len(df),
        "sample_size_requested": sample_size,
        "sampled_pdf_pages": df["pdf_page"].tolist(),
        "sample_status_counts": (
            df["detector_match_status"].value_counts().to_dict()
        ),
        "section_counts": df["section_group"].value_counts().to_dict(),
    }


def validate_missed_risk_audit(
    csv_path="data/missed_risk_audit.csv",
):
    """Validate audit schema and review outcomes."""

    path = Path(csv_path)

    if not path.is_file():
        return {
            "valid": False,
            "rows": 0,
            "errors": [f"Audit file not found: {path}"],
            "pending_reviews": 0,
            "reviewed_records": 0,
        }

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            keep_default_na=False,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        return {
            "valid": False,
            "rows": 0,
            "errors": [f"Could not read audit CSV: {exc}"],
            "pending_reviews": 0,
            "reviewed_records": 0,
        }

    errors = []
    missing_columns = [
        column for column in AUDIT_COLUMNS
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
            "pending_reviews": 0,
            "reviewed_records": 0,
        }

    if df.empty:
        errors.append("Audit CSV has no rows.")

    required_fields = [
        "company",
        "financial_year",
        "source_pdf",
        "pdf_page",
        "section_group",
        "detector_findings",
        "detector_match_status",
        "manual_review_status",
    ]

    for column in required_fields:
        for index in df.index[
            df[column].astype(str).str.strip().eq("")
        ]:
            errors.append(
                f"Row {index + 2}: '{column}' is blank."
            )

    page_numbers = pd.to_numeric(df["pdf_page"], errors="coerce")
    invalid_pages = (
        page_numbers.isna()
        | (page_numbers <= 0)
        | ((page_numbers % 1) != 0)
    )

    for index in df.index[invalid_pages]:
        errors.append(
            f"Row {index + 2}: pdf_page must be a positive integer."
        )

    statuses = df["manual_review_status"].astype(str).str.strip()
    match_statuses = df["detector_match_status"].astype(str).str.strip()

    manual_values = (
        df["manual_risk_found"].astype(str).str.strip().str.lower()
    )
    missed_values = (
        df["detector_missed_event"].astype(str).str.strip().str.lower()
    )

    for index, status in statuses.items():
        if status not in ALLOWED_REVIEW_STATUSES:
            errors.append(
                f"Row {index + 2}: invalid manual_review_status "
                f"'{status}'."
            )

    for index, status in match_statuses.items():
        if status not in ALLOWED_MATCH_STATUSES:
            errors.append(
                f"Row {index + 2}: invalid detector_match_status "
                f"'{status}'."
            )

        categories = df.loc[
            index, "detector_match_categories"
        ].strip()
        suppressed = df.loc[
            index, "category_limit_suppressed"
        ].strip()

        if status == "no_rule_match" and categories:
            errors.append(
                f"Row {index + 2}: no_rule_match cannot have "
                "detector_match_categories."
            )

        if status in {
            "suppressed_by_category_limit",
            "matched_but_not_reported",
        } and not categories:
            errors.append(
                f"Row {index + 2}: matched-but-not-reported status "
                "requires detector_match_categories."
            )

        if status == "suppressed_by_category_limit" and not suppressed:
            errors.append(
                f"Row {index + 2}: category-limit suppression "
                "requires category_limit_suppressed."
            )

        if status != "suppressed_by_category_limit" and suppressed:
            errors.append(
                f"Row {index + 2}: category_limit_suppressed must be "
                "blank unless suppression is the reported status."
            )

    for index, value in manual_values.items():
        if value not in ALLOWED_MANUAL_RISK_VALUES:
            errors.append(
                f"Row {index + 2}: invalid manual_risk_found '{value}'."
            )

    for index, value in missed_values.items():
        if value not in ALLOWED_MISSED_EVENT_VALUES:
            errors.append(
                f"Row {index + 2}: invalid detector_missed_event "
                f"'{value}'."
            )

    # Validate each reviewed row independently. Use stripped strings
    # so blank CSV cells cannot be mistaken for valid evidence.
    for index, row in df.iterrows():
        status = str(row["manual_review_status"]).strip()
        if status != "reviewed":
            continue

        outcome = str(row["manual_risk_found"]).strip().lower()
        missed = str(row["detector_missed_event"]).strip().lower()

        if outcome not in {"yes", "no", "uncertain"}:
            errors.append(
                f"Row {index + 2}: reviewed record requires "
                "manual_risk_found = yes, no, or uncertain."
            )
            continue

        if outcome == "yes":
            category = str(row["manual_event_category"]).strip()
            description = str(row["event_description"]).strip()

            if not category:
                errors.append(
                    f"Row {index + 2}: risk found but event category "
                    "is missing."
                )

            if not description:
                errors.append(
                    f"Row {index + 2}: risk found but event description "
                    "is missing."
                )

            if missed != "yes":
                errors.append(
                    f"Row {index + 2}: manually found risk on a page "
                    "without a displayed finding requires "
                    "detector_missed_event = yes."
                )

        elif outcome == "no" and missed != "no":
            errors.append(
                f"Row {index + 2}: when no risk is found, "
                "detector_missed_event should be 'no'."
            )

        elif outcome == "uncertain" and missed != "uncertain":
            errors.append(
                f"Row {index + 2}: uncertain outcome requires "
                "detector_missed_event = 'uncertain'."
            )

    duplicate_key = [
        "company",
        "financial_year",
        "source_pdf",
        "pdf_page",
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    for index in df.index[duplicate_mask]:
        errors.append(
            f"Row {index + 2}: duplicate audit page."
        )

    return {
        "valid": not errors,
        "rows": len(df),
        "errors": errors,
        "pending_reviews": int(
            statuses.eq("pending_manual_review").sum()
        ),
        "reviewed_records": int(statuses.eq("reviewed").sum()),
        "match_status_counts": match_statuses.value_counts().to_dict(),
    }


def main():
    """Generate an audit sample from a local annual report."""

    parser = argparse.ArgumentParser(
        description=(
            "Create an audit sample distinguishing pages with no rule "
            "matches from pages whose matches were not displayed."
        )
    )

    parser.add_argument(
        "--pdf",
        default="data/reports/TCS_FY2024-25.pdf",
    )
    parser.add_argument(
        "--output",
        default="data/missed_risk_audit.csv",
    )
    parser.add_argument("--company", default="TCS")
    parser.add_argument("--financial-year", default="2024-25")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = extract_pdf_pages(pdf_path.read_bytes())

    summary = build_missed_risk_audit_from_pages(
        pages=pages,
        output_csv=args.output,
        company=args.company,
        financial_year=args.financial_year,
        source_pdf=pdf_path.name,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    print("\n========== MISSED-RISK AUDIT ==========")
    print("PDF:", pdf_path)
    print("Total PDF pages:", summary["total_pdf_pages"])
    print(
        "Pages with displayed findings:",
        summary["pages_with_reported_findings"],
    )
    print(
        "Eligible pages without displayed findings:",
        summary["eligible_pages_without_reported_findings"],
    )
    print(
        "Pages with no rule match:",
        summary["no_rule_match_pages_available"],
    )
    print(
        "Pages suppressed by category limit:",
        summary["suppressed_pages_available"],
    )
    print("Rows written:", summary["rows_written"])
    print("Sample status counts:", summary["sample_status_counts"])
    print("Section counts:", summary["section_counts"])
    print("Sampled PDF pages:", summary["sampled_pdf_pages"])
    print("Output:", summary["output_csv"])
    print("\nAll rows remain pending manual review.")


if __name__ == "__main__":
    main()
