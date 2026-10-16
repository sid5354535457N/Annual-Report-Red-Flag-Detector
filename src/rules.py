
import re
from difflib import SequenceMatcher


# Initial research heuristics, not validated probabilities of fraud,
# financial distress, or stock-price declines.
RULES = [
    {
        "key": "qualified_opinion",
        "category": "Qualified or adverse audit opinion",
        "type": "risk_signal",
        "weight": 35,
        "patterns": [
            r"\bqualified opinion\b",
            r"\badverse opinion\b",
            r"\bdisclaimer of opinion\b",
            r"\bexcept for the effects of\b",
        ],
    },
    {
        "key": "going_concern",
        "category": "Going-concern uncertainty",
        "type": "risk_signal",
        "weight": 35,
        "patterns": [
            r"\bmaterial uncertainty\b.{0,100}\bgoing concern\b",
            r"\bgoing concern\b.{0,100}\bmaterial uncertainty\b",
            r"\bsubstantial doubt\b.{0,80}\bgoing concern\b",
            r"\bunable to continue as a going concern\b",
            r"\bgoing concern assumption is inappropriate\b",
        ],
    },
    {
        "key": "fraud_or_irregularity",
        "category": "Possible fraud or irregularity",
        "type": "risk_signal",
        "weight": 30,
        "patterns": [
            r"\bsuspected fraud\b",
            r"\bfraudulent transactions?\b",
            r"\bfraud was detected\b",
            r"\bfraud has been detected\b",
            r"\bfraud has been reported\b",
        ],
    },
    {
        "key": "adverse_legal_judgment",
        "category": "Adverse legal judgment or damages",
        "type": "risk_signal",
        "weight": 25,
        "patterns": [
            (
                r"\b(?:the\s+)?court\s+ordered\s+that\s+"
                r"the\s+company\s+is\s+liable\s+to\b"
            ),
            (
                r"\b(?:the\s+)?(?:trial\s+)?court\s+"
                r"(?:found|held)\s+that\s+the\s+company\s+"
                r"is\s+liable\s+to\b"
            ),
            (
                r"\bthe\s+company\s+is\s+liable\s+to\b"
                r"(?=.{0,250}\b(?:damages|interest)\b)"
            ),
        ],
    },
    {
        "key": "litigation_allegation",
        "category": "Legal allegation requiring contextual review",
        "type": "review_item",
        "weight": 0,
        "patterns": [
            (
                r"\b(?:alleged|alleging|alleges|allegation\s+of)\b"
                r".{0,120}\bmisappropriation\b"
            ),
            (
                r"\bmisappropriation\b.{0,120}"
                r"\b(?:alleged|alleging|claim|claims|court|"
                r"lawsuit|litigation|proceedings?)\b"
            ),
        ],
    },
    {
        "key": "internal_control_weakness",
        "category": "Internal control weakness",
        "type": "risk_signal",
        "weight": 25,
        "patterns": [
            (
                r"\bmaterial weakness(?:es)? in "
                r"(?:the\s+)?internal (?:financial\s+)?controls?\b"
            ),
            (
                r"\bsignificant deficiencies in "
                r"(?:the\s+)?internal controls?\b"
            ),
            (
                r"\badverse opinion on "
                r"(?:the\s+)?internal financial controls?\b"
            ),
            r"\binternal controls? were not operating effectively\b",
        ],
    },
    {
        "key": "auditor_resignation",
        "category": "Auditor resignation",
        "type": "risk_signal",
        "weight": 20,
        "patterns": [
            r"\bstatutory auditor has resigned\b",
            r"\bresignation of the statutory auditor\b",
            r"\bauditor resigned during the year\b",
        ],
    },
    {
        "key": "regulatory_noncompliance",
        "category": "Potential statutory non-compliance",
        "type": "risk_signal",
        "weight": 20,
        "patterns": [
            r"\bfailure to deposit statutory dues\b",
            (
                r"\bstatutory dues\b.{0,160}"
                r"\b(?:not deposited|remaining unpaid|"
                r"outstanding for more than six months|"
                r"outstanding for over six months)\b"
            ),
        ],
    },
    {
        "key": "quantified_tax_dispute",
        "category": "Quantified tax dispute requiring review",
        "type": "review_item",
        "weight": 0,
        "patterns": [
            (
                r"\b(?:direct|indirect)\s+tax\s+matters?\b"
                r"(?=(?:(?!\b(?:direct|indirect)\s+tax\s+matters?\b)"
                r".){0,1400}"
                r"\bdemands?\s+amounting\s+to\b"
                r".{0,150}"
                r"\b\d[\d,]*(?:\s*\.\s*\d+)?\s+crore\b)"
            ),
        ],
    },
    {
        "key": "other_disputed_claims",
        "category": "Other quantified disputed claims requiring review",
        "type": "review_item",
        "weight": 0,
        "patterns": [
            (
                r"\bclaims?\s+aggregating\b"
                r"(?=.{0,300}\b\d[\d,]*(?:\s*\.\s*\d+)?\s+crore\b"
                r".{0,220}\b(?:not\s+been\s+acknowledged\s+as\s+debts|"
                r"contested|disputed)\b)"
            ),
        ],
    },
    {
        "key": "emphasis_of_matter",
        "category": "Emphasis-of-matter disclosure",
        "type": "review_item",
        "weight": 0,
        "patterns": [
            r"\bemphasis of matter\b",
        ],
    },
    {
        "key": "related_party",
        "category": "Related-party transaction disclosure",
        "type": "disclosure",
        "weight": 0,
        "patterns": [
            r"\brelated[- ]party transactions?\b",
            r"\btransactions? with related parties\b",
        ],
    },
    {
        "key": "contingent_liability",
        "category": "Contingent-liability disclosure",
        "type": "disclosure",
        "weight": 0,
        "patterns": [
            r"\bcontingent liabilities\b",
            r"\bcontingent liability\b",
        ],
    },
]


NEGATION_PATTERNS = {
    "qualified_opinion": [
        r"\bnot a qualified opinion\b",
        r"\bno qualified opinion\b",
        r"\bnot an adverse opinion\b",
        r"\bno adverse opinion\b",
    ],
    "going_concern": [
        r"\bno material uncertainty\b.{0,100}\bgoing concern\b",
        r"\bno significant doubt\b.{0,100}\bgoing concern\b",
    ],
    "fraud_or_irregularity": [
        r"\bno\b.{0,30}\b(?:fraud|misappropriation)\b",
        (
            r"\b(?:fraud|misappropriation)\b.{0,60}"
            r"\b(?:not detected|not identified|not reported|"
            r"has not been detected|has not been reported|"
            r"was not detected|was not reported)\b"
        ),
    ],
    "internal_control_weakness": [
        r"\bno material weaknesses?\b.{0,100}\binternal\b",
        r"\bno significant deficiencies\b.{0,100}\binternal\b",
    ],
}


EXCLUSION_PATTERNS = {
    "internal_control_weakness": [
        (
            r"\bsignificant deficiencies in internal control\b"
            r".{0,500}\bwe identify during our audit\b"
        ),
        (
            r"\bsignificant deficiencies in internal control\b"
            r".{0,500}\bwe identify\b"
        ),
    ],
}


LEGAL_EVENT_KEYS = {
    "adverse_legal_judgment",
    "litigation_allegation",
}


# Allow OCR spacing in headings such as "Other c laims".
# Stop an evidence excerpt before the next section.
SECTION_BOUNDARY_PATTERN = (
    r"\b(?:direct|indirect)\s+tax\s+matters?\b"
    r"|\bother\s+c?\s*laims\b"
    r"|\bletter\s+of\s+comfort\b"
    r"|\bguarantees\s+and\s+letter\s+of\s+comfort\b"
)


def normalize_context(text: str) -> str:
    """Normalize whitespace in extracted PDF text."""

    return re.sub(r"\s+", " ", text).strip()


def normalize_for_similarity(text: str) -> str:
    """Normalize evidence for approximate duplicate comparison."""

    text = normalize_context(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def is_negated(
    category_key: str,
    text: str,
    start: int,
    end: int,
) -> bool:
    """Check nearby wording for a simple negation."""

    left = max(0, start - 150)
    right = min(len(text), end + 150)

    context = normalize_context(text[left:right]).lower()
    patterns = NEGATION_PATTERNS.get(category_key, [])

    return any(
        re.search(
            pattern,
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in patterns
    )


def is_excluded_match(
    category_key: str,
    text: str,
    start: int,
    end: int,
) -> bool:
    """Exclude recognized generic auditor boilerplate."""

    left = max(0, start - 400)
    right = min(len(text), end + 400)

    context = normalize_context(text[left:right])
    patterns = EXCLUSION_PATTERNS.get(category_key, [])

    return any(
        re.search(
            pattern,
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in patterns
    )


def make_evidence(
    text: str,
    start: int,
    end: int,
    category_key: str,
) -> str:
    """Build readable evidence excerpts with category-specific context."""

    if category_key == "quantified_tax_dispute":
        left = start
        remainder = text[end:]

        boundary = re.search(
            SECTION_BOUNDARY_PATTERN,
            remainder,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if boundary:
            right = end + boundary.start()
        else:
            right = min(len(text), start + 1500)

        right = min(right, start + 1500)
        return normalize_context(text[left:right])

    if category_key == "other_disputed_claims":
        # Stop after the status phrase so the following legal matter
        # does not get included in this claims excerpt.
        remainder = text[end:]

        status = re.search(
            r"\b(?:not\s+been\s+acknowledged\s+as\s+debts|"
            r"contested|disputed)\b",
            remainder,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if status:
            right = end + status.end()
        else:
            right = min(len(text), end + 300)

        return normalize_context(text[start:right])

    if category_key == "adverse_legal_judgment":
        left = max(0, start - 200)
        right = min(len(text), end + 500)
    else:
        left = max(0, start - 100)
        right = min(len(text), end + 100)

    excerpt = normalize_context(text[left:right])

    if left > 0:
        excerpt = "..." + excerpt

    if right < len(text):
        excerpt += "..."

    return excerpt


def find_duplicate_legal_event(
    category_key: str,
    evidence: str,
    findings: list[dict],
) -> dict | None:
    """
    Find a near-identical legal passage already detected elsewhere.

    Similarity matching is heuristic. It cannot guarantee that two
    passages describe the same underlying event.
    """

    if category_key not in LEGAL_EVENT_KEYS:
        return None

    current = normalize_for_similarity(evidence)

    if len(current) < 50:
        return None

    for existing in findings:
        if existing.get("category_key") != category_key:
            continue

        previous = normalize_for_similarity(
            existing.get("evidence", "")
        )

        if not previous:
            continue

        similarity = SequenceMatcher(
            None,
            current,
            previous,
        ).ratio()

        if similarity >= 0.80:
            return existing

    return None


def detect_red_flags(pages: list[dict]) -> dict:
    """
    Detect potential red flags and review items in extracted PDF pages.

    Each page should contain:
        {"page": 1, "text": "Extracted report text"}

    Finding types:
        risk_signal: Potentially adverse language requiring verification.
        review_item: A quantified exposure or disclosure needing review.
        disclosure: A disclosure not scored by itself.

    The score is heuristic, not a probability of financial distress,
    fraud, or future stock-price movements.
    """

    findings = []
    seen = set()
    count_by_category = {}

    for page_data in pages:
        page_number = page_data.get("page")
        text = page_data.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        for rule in RULES:
            for pattern in rule["patterns"]:
                for match in re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    identity = (rule["key"], page_number)

                    if identity in seen:
                        continue

                    if is_excluded_match(
                        rule["key"],
                        text,
                        match.start(),
                        match.end(),
                    ):
                        continue

                    count = count_by_category.get(rule["key"], 0)

                    if count >= 3:
                        continue

                    if is_negated(
                        rule["key"],
                        text,
                        match.start(),
                        match.end(),
                    ):
                        continue

                    evidence = make_evidence(
                        text,
                        match.start(),
                        match.end(),
                        rule["key"],
                    )

                    duplicate = find_duplicate_legal_event(
                        rule["key"],
                        evidence,
                        findings,
                    )

                    if duplicate is not None:
                        additional_pages = duplicate.setdefault(
                            "also_reported_on_pages", []
                        )

                        if (
                            page_number != duplicate.get("page")
                            and page_number not in additional_pages
                        ):
                            additional_pages.append(page_number)

                        seen.add(identity)
                        continue

                    finding = {
                        "category_key": rule["key"],
                        "category": rule["category"],
                        "type": rule["type"],
                        "weight": rule["weight"],
                        "page": page_number,
                        "matched_term": match.group(0),
                        "evidence": evidence,
                    }

                    findings.append(finding)
                    seen.add(identity)
                    count_by_category[rule["key"]] = count + 1

    # Each risk category contributes its weight at most once.
    risk_weights = {
        rule["key"]: rule["weight"]
        for rule in RULES
        if rule["type"] == "risk_signal"
    }

    detected_categories = {
        finding["category_key"]
        for finding in findings
        if finding["type"] == "risk_signal"
    }

    score = min(
        100,
        sum(risk_weights[key] for key in detected_categories),
    )

    if score >= 50:
        priority = "High review priority"
    elif score >= 25:
        priority = "Moderate review priority"
    elif score > 0:
        priority = "Low review priority"
    else:
        priority = "No configured risk terms detected"

    return {
        "score": score,
        "priority": priority,
        "findings": findings,
        "disclaimer": (
            "This is an experimental rule-based score, not a "
            "validated financial-risk probability or investment signal. "
            "Detected language requires contextual review."
        ),
    }
