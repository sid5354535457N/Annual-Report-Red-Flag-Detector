import hashlib

import pandas as pd
import streamlit as st

from src.pdf_parser import (
    PDFExtractionError,
    extract_pdf_pages,
)
from src.rules import detect_red_flags
from src.gemini_assistant import (
    GeminiAssistantError,
    ask_gemini,
    gemini_is_configured,
)


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Annual Report Red-Flag Detector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Annual Report Red-Flag Detector")

st.markdown(
    """
    Analyze annual reports to identify potential audit and
    financial-reporting warning signs, inspect source evidence,
    and ask an AI assistant to explain the results.
    """
)


# --------------------------------------------------
# 2. Helper: render retrieved evidence
# --------------------------------------------------

def render_evidence(evidence: list[dict]) -> None:
    """Display the retrieved source passages used for an answer."""

    with st.expander(
        f"Source evidence ({len(evidence)} passages)"
    ):
        if not evidence:
            st.info(
                "No relevant passages were retrieved for this answer. "
                "This does not establish that the topic is absent "
                "from the complete report."
            )
            return

        for index, passage in enumerate(evidence, start=1):
            page_number = passage.get("page", "Unknown")
            text = passage.get("text", "")
            similarity = passage.get("similarity")

            st.markdown(
                f"**Passage {index} — PDF page {page_number}**"
            )

            if similarity is not None:
                st.caption(
                    f"TF-IDF similarity: {similarity:.3f}. "
                    "This is a text-matching score, not factual confidence."
                )

            st.write(text)


# --------------------------------------------------
# 3. Sidebar and input
# --------------------------------------------------

st.sidebar.header("Document Analysis")

demo_mode = st.sidebar.checkbox(
    "Use fictional demo data",
    value=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload an annual-report PDF",
    type=["pdf"],
    disabled=demo_mode,
)

st.sidebar.divider()

st.sidebar.caption(
    "Research prototype only. The rule score is not a validated "
    "probability of fraud, financial distress, or future losses."
)

if gemini_is_configured():
    st.sidebar.success("Gemini API key detected")
else:
    st.sidebar.warning("Gemini API key not configured")


# --------------------------------------------------
# 4. Fictional demo pages
# --------------------------------------------------

demo_pages = [
    {
        "page": 2,
        "text": (
            "Independent Auditor's Report. "
            "We have expressed a qualified opinion on the "
            "financial statements for the fictional company."
        ),
    },
    {
        "page": 11,
        "text": (
            "Material uncertainty related to going concern. "
            "These conditions indicate that a material uncertainty "
            "exists about the company's ability to continue "
            "as a going concern."
        ),
    },
    {
        "page": 16,
        "text": (
            "Note 32: Related-party transactions. "
            "Transactions with related parties are disclosed "
            "in the accompanying financial statements. "
            "Emphasis of Matter: See Note 40."
        ),
    },
]


# --------------------------------------------------
# 5. Read the uploaded document or demo data
# --------------------------------------------------

pages = None
source_name = None

if demo_mode:
    pages = demo_pages
    source_name = "Fictional demonstration report"

elif uploaded_file is None:
    st.info(
        "Enable fictional demo data or upload a PDF to begin analysis."
    )
    st.stop()

else:
    try:
        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pdf_pages(pdf_bytes)
        source_name = uploaded_file.name

    except PDFExtractionError as error:
        st.error(str(error))
        st.stop()


# --------------------------------------------------
# 6. Run the red-flag detector
# --------------------------------------------------

result = detect_red_flags(pages)
findings = result["findings"]

risk_signal_count = sum(
    finding.get("type") == "risk_signal"
    for finding in findings
)


# --------------------------------------------------
# 7. Display analysis metrics
# --------------------------------------------------

st.subheader("Analysis Results")
st.caption(f"Document: {source_name}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Experimental rule score",
        f"{result['score']} / 100",
    )

with col2:
    st.metric(
        "Review priority",
        result["priority"],
    )

with col3:
    st.metric(
        "Risk-signal matches",
        risk_signal_count,
    )

st.info(result["disclaimer"])


# --------------------------------------------------
# 8. Findings table
# --------------------------------------------------

st.subheader("Detected Findings")

if not findings:
    st.warning(
        "No configured phrases were detected. "
        "This does not establish that the company is low risk."
    )

else:
    table_data = []

    for finding in findings:
        table_data.append(
            {
                "Category": finding["category"],
                "Type": finding["type"].replace("_", " ").title(),
                "PDF Page": finding["page"],
                "Matched Phrase": finding["matched_term"],
                "Category Weight": finding["weight"],
            }
        )

    findings_df = pd.DataFrame(table_data)

    st.dataframe(
        findings_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Category weights are experimental heuristics. "
        "Disclosure matches do not automatically add risk points."
    )


# --------------------------------------------------
# 9. Supporting evidence for detected findings
# --------------------------------------------------

st.subheader("Evidence and Source Pages")

if findings:
    for index, finding in enumerate(findings, start=1):
        heading = (
            f"{index}. Page {finding['page']} — "
            f"{finding['category']}"
        )

        with st.expander(heading):
            st.write(
                f"**Matched phrase:** {finding['matched_term']}"
            )

            st.write(
                "**Finding type:** "
                f"{finding['type'].replace('_', ' ').title()}"
            )

            st.write("**Supporting passage**")
            st.write(finding["evidence"])

            st.caption(
                "Verify this passage in the original PDF before "
                "interpreting it as a financial warning."
            )


# --------------------------------------------------
# 10. Extracted report text
# --------------------------------------------------

st.subheader("Extracted Document Text")

available_pages = [
    page for page in pages
    if page.get("text", "").strip()
]

if available_pages:
    selected_page_number = st.selectbox(
        "Select a PDF page",
        options=[
            page["page"] for page in available_pages
        ],
    )

    selected_page = next(
        page for page in available_pages
        if page["page"] == selected_page_number
    )

    st.text_area(
        "Extracted text",
        value=selected_page["text"],
        height=260,
        disabled=True,
    )

else:
    st.warning("No pages contain extractable text.")


# --------------------------------------------------
# 11. Gemini AI Research Assistant
# --------------------------------------------------

st.divider()
st.subheader("💬 AI Research Assistant")

st.write(
    "Ask questions about the report, detected findings, "
    "supporting evidence, or the limitations of the analysis."
)

st.caption(
    "Gemini receives the findings and the most relevant passages "
    "retrieved from the extracted document. It does not automatically "
    "receive the complete contents of every page."
)

if not gemini_is_configured():
    st.warning(
        "Gemini is not configured. Add GEMINI_API_KEY to your .env "
        "file and restart Streamlit before submitting a question."
    )


# Create a fingerprint so that conversation history is cleared
# when a different document is selected.
document_content = (
    source_name
    + "\n"
    + "\n".join(
        f"{page['page']}:{page['text']}"
        for page in pages
    )
)

document_fingerprint = hashlib.sha256(
    document_content.encode("utf-8")
).hexdigest()

if (
    st.session_state.get("gemini_document_fingerprint")
    != document_fingerprint
):
    st.session_state["gemini_document_fingerprint"] = (
        document_fingerprint
    )
    st.session_state["gemini_messages"] = []


# Welcome message when the chat is empty.
if not st.session_state["gemini_messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I can explain the findings in this report. "
            "Try asking: **Why was this company flagged?**"
        )


# Render earlier messages and their associated evidence.
for message in st.session_state["gemini_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "evidence" in message
        ):
            render_evidence(message["evidence"])


# --------------------------------------------------
# 12. Handle a new user question
# --------------------------------------------------

question = st.chat_input(
    "Ask about the report or its findings..."
)

if question:
    # Save the user's message.
    st.session_state["gemini_messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    evidence_for_answer = []

    with st.chat_message("assistant"):
        with st.spinner(
            "Retrieving evidence and generating an explanation..."
        ):
            try:
                answer = ask_gemini(
                    question=question,
                    pages=pages,
                    result=result,
                    source_name=source_name,
                    chat_history=(
                        st.session_state["gemini_messages"][:-1]
                    ),
                    evidence_out=evidence_for_answer,
                )

            except GeminiAssistantError as error:
                answer = f"**Assistant error:** {error}"

            except Exception:
                answer = (
                    "An unexpected error occurred. Check the VS Code "
                    "terminal for details, then try again."
                )

        st.markdown(answer)

        # Display the same passages supplied to Gemini.
        render_evidence(evidence_for_answer)

    # Preserve both the answer and its evidence in session history.
    st.session_state["gemini_messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "evidence": evidence_for_answer,
        }
    )


# --------------------------------------------------
# 13. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Annual Report Red-Flag Detector | "
    "Experimental rule-based scoring | "
    "TF-IDF evidence retrieval | Gemini explanations"
)