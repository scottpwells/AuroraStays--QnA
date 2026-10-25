"""Staff-facing hotel policy questions, using the evaluated backend defaults."""

import ast
import os
import re

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:7860").strip().rstrip("/")
REQUEST_TIMEOUT = (5, 90)


def policy_markdown(text):
    """Display policy currency literally instead of treating dollar pairs as math."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def request_answer(query, include_sources, k, temperature, max_tokens):
    endpoint = "/v1/answer_with_relevant_chunks" if include_sources else "/v1/answer"
    response = requests.post(
        BACKEND_URL + endpoint,
        json={"query": query, "k": k, "model_name": "gpt-4o-mini",
              "temperature": temperature, "top_p": 1.0, "max_tokens": max_tokens},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("Invalid answer response.")
    chunks = result.get("relevant_chunks", [])
    if include_sources and ("relevant_chunks" not in result or not isinstance(chunks, list)
                            or not all(isinstance(chunk, str) for chunk in chunks)):
        raise ValueError("Invalid policy excerpts.")
    return {"query": query, "answer": result["answer"].strip(),
            "chunks": chunks if include_sources else [], "include_sources": include_sources}


def policy_excerpt(chunk):
    content, _, raw_metadata = chunk.partition("\n\nMetadata:\n")
    content = content.removeprefix("Content:\n").strip()
    try:
        metadata = ast.literal_eval(raw_metadata) if raw_metadata else {}
    except (SyntaxError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    heading = metadata.get("section", "")
    if content.startswith("Section: "):
        first, _, content = content.partition("\n")
        heading = heading or first.removeprefix("Section: ")
    sources = metadata.get("sources") or [metadata.get("source", "")]
    if not isinstance(sources, list):
        sources = []
    filenames = list(dict.fromkeys(source.replace("\\", "/").rsplit("/", 1)[-1]
                                   for source in sources if isinstance(source, str) and source))
    return content.strip(), heading, filenames


st.set_page_config(page_title="Aurora Stays | Policy desk", page_icon="🏨", layout="centered")
st.title("🏨 Aurora Stays")
st.write("Answers to your hotel policy questions.")

with st.sidebar:
    st.header("Policy desk")
    st.caption("Ask about bookings, rooms, amenities, fees, or guest policies.")
    with st.expander("Advanced answer settings", expanded=False):
        st.caption("The defaults match the evaluated configuration.")
        k = st.slider("Policy excerpts to retrieve", 1, 10, 4, key="retrieval_k")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="temperature")
        max_tokens = st.slider("Answer token limit", 64, 2048, 512, 64, key="max_tokens")

with st.form("policy_question", clear_on_submit=False):
    query = st.text_area(
        "Your question", placeholder="What are the pet fees at ASH-CHI Loop?",
        height=110, max_chars=8000, key="question",
    )
    response_mode = st.radio(
        "Response mode", ["Answer only", "Answer with sources"],
        horizontal=True, key="response_mode",
    )
    submitted = st.form_submit_button("Ask", type="primary", width="stretch")

st.session_state.setdefault("policy_result", None)
if submitted:
    st.session_state["policy_result"] = None
    if not query.strip():
        st.warning("Please enter a hotel policy question.")
    else:
        try:
            with st.spinner("Checking the policy documents…"):
                st.session_state["policy_result"] = request_answer(
                    query.strip(), response_mode == "Answer with sources", k, temperature, max_tokens,
                )
        except requests.Timeout:
            st.error("The answer is taking longer than expected. Please try again.")
        except requests.HTTPError:
            st.error("The policy service could not answer this request. Please try again shortly.")
        except requests.RequestException:
            st.error("The policy service is unavailable. Please try again shortly.")
        except (ValueError, TypeError):
            st.error("The policy service returned an unexpected response. Please try again.")

result = st.session_state["policy_result"]
if result:
    st.divider()
    st.subheader("Answer")
    st.caption(result["query"])
    st.markdown(policy_markdown(result["answer"]))
    if result["include_sources"]:
        st.subheader("Supporting policy excerpts")
        if not result["chunks"]:
            st.info("No supporting excerpts were returned for this question.")
        for number, chunk in enumerate(result["chunks"], 1):
            content, heading, sources = policy_excerpt(chunk)
            with st.expander(heading or f"Policy excerpt {number}"):
                st.markdown(policy_markdown(content).replace("\n", "  \n"))
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
