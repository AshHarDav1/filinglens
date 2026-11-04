"""FilingLens demo UI. A thin client: it only talks to the HTTP API, never to the agent directly.

uv run uvicorn filinglens.api.main:app        # first, the API
uv run streamlit run ui/app.py                # then the UI, at http://localhost:8501
"""

import httpx
import streamlit as st

from filinglens.api.client import FilingLensClient, FilingLensError
from filinglens.api.schemas import AskResponse
from filinglens.config import get_settings

EXAMPLES = {
    "PepsiCo EBITDA margin": (
        "What is the FY2022 unadjusted EBITDA % margin for PepsiCo? Calculate unadjusted EBITDA "
        "using unadjusted operating income and D&A (from cash flow statement)."
    ),
    "Boeing legal battles": (
        "Has Boeing reported any materially important ongoing legal battles from FY2022?"
    ),
    "Should I buy AMD?": "Should I buy AMD stock based on its FY2022 results?",
}
OUTCOME = {"answered": "Answered", "not_found": "Not found in the filings", "refused": "Refused"}


def md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; financial answers are full of dollar signs."""
    return text.replace("$", r"\$")


def show_answer(entry: dict) -> None:
    if entry["error"]:
        st.error(entry["error"])
        return
    result = AskResponse.model_validate(entry["result"])
    st.markdown(md(result.answer))
    seconds = f"{result.latency_ms / 1000:.1f} s"
    if result.outcome == "answered":
        check = "✅ verified" if result.verified else "⚠️ not verified"
        st.caption(f"{check} · {result.rounds} tool rounds · {seconds}")
    else:
        st.caption(f"{OUTCOME[result.outcome]} · {seconds}")
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for s in result.sources:
                link = f" · [SEC filing]({s.url})" if s.url else ""
                st.markdown(md(f"**[{s.n}]** {s.label}{link}"))
    if entry["steps"]:
        with st.expander("Agent steps"):
            for line in entry["steps"]:
                st.text(line)
    if result.tool_calls:
        with st.expander("Tool calls"):
            st.json([c.model_dump() for c in result.tool_calls])


def run(client: FilingLensClient, question: str) -> dict:
    entry: dict = {"question": question, "steps": [], "result": None, "error": None}
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        with st.status("Sending the question…", expanded=True) as status:
            try:
                for event in client.ask_stream(question):
                    if event.type == "step":
                        label = event.data.get("label") or event.data["node"]
                        line = f"{event.data['elapsed_ms'] / 1000:5.1f} s  {label}"
                        entry["steps"].append(line)
                        status.write(md(line))
                        status.update(label=md(label))
                    elif event.type == "result":
                        entry["result"] = event.data
                    else:
                        entry["error"] = event.data.get("detail", "The API reported an error.")
            except FilingLensError as error:
                entry["error"] = f"API error {error.status}: {error.detail}"
            except httpx.HTTPError:
                entry["error"] = (
                    f"Can't reach the API at {get_settings().api_url}. "
                    "Start it with `uv run uvicorn filinglens.api.main:app`."
                )
            if entry["error"] or entry["result"] is None:
                entry["error"] = entry["error"] or "The stream ended without a result."
                status.update(label="Failed", state="error", expanded=False)
            else:
                done = entry["result"]["latency_ms"] / 1000
                status.update(label=f"Done in {done:.1f} s", state="complete", expanded=False)
        show_answer(entry)
    return entry


def sidebar(client: FilingLensClient) -> None:
    with st.sidebar:
        st.subheader("Status")
        try:
            health = client.health()
        except httpx.HTTPError:
            st.error(f"API not reachable at {get_settings().api_url}")
            return
        if health.status == "ok":
            st.success(f"API ok · {health.companies} companies · {health.filings} 10-Ks")
        else:
            st.error("API up, database unreachable")
        st.caption(
            f"Answers: {health.models.answer} · guard: {health.models.guard} · API {health.version}"
        )
        st.caption("Covers AMD, Amazon, Adobe, American Express, Boeing, 3M, Microsoft, "
                   "Netflix, PepsiCo and Verizon.")  # fmt: skip
        if st.button("Clear chat"):
            st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="FilingLens", page_icon="📄")
    client = st.session_state.setdefault("client", None) or FilingLensClient()
    st.session_state.client = client
    history: list[dict] = st.session_state.setdefault("history", [])

    st.title("FilingLens")
    st.caption(
        "Questions about US companies' 10-K filings, answered with exact XBRL numbers and "
        "cited filing text. Not investment advice."
    )
    sidebar(client)

    for entry in history:
        with st.chat_message("user"):
            st.markdown(md(entry["question"]))
        with st.chat_message("assistant"):
            show_answer(entry)

    question = None
    if not history:
        st.write("Try one:")
        for column, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items(), strict=True):
            if column.button(label, use_container_width=True):
                question = text
    question = st.chat_input("Ask about a 10-K…") or question
    if question:
        history.append(run(client, question))


main()
