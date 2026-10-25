"""Flask endpoints using the notebook's evaluated hotel-policy pipeline."""

import hashlib
import json
import math
from pathlib import Path

from flask import Flask, jsonify, request
from rag_utils import rag, retrieve

BACKEND_DIR = Path(__file__).resolve().parent


def load_vectorstore():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    manifest = json.loads((BACKEND_DIR / "deployment_manifest.json").read_text())
    # Verify our own exported index before loading its pickle-based docstore.
    for relative, expected in manifest["artifact_hashes"].items():
        if hashlib.sha256((BACKEND_DIR / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Backend artifact does not match its manifest: {relative}")
    embeddings = HuggingFaceEmbeddings(
        model_name=manifest["embedding_model"],
        model_kwargs={"device": "cpu", "revision": manifest["embedding_revision"]},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(BACKEND_DIR / "faiss_index"), embeddings,
        allow_dangerous_deserialization=True,
    )
    if (vectorstore.index.ntotal, vectorstore.index.d) != (manifest["vectors"], manifest["dimensions"]):
        raise RuntimeError("Backend index dimensions/count differ from the evaluated index.")
    return vectorstore


def request_parameters():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object containing query.")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text.")
    if len(query) > 8000:
        raise ValueError("query must contain at most 8000 characters.")
    k = data.get("k", 4)
    if type(k) is not int or not 1 <= k <= 16:
        raise ValueError("k must be an integer from 1 to 16.")
    model_name = data.get("model_name", "gpt-4o-mini")
    if not isinstance(model_name, str) or not model_name.strip() or len(model_name) > 128:
        raise ValueError("model_name must be nonempty text of at most 128 characters.")
    parameters = {"query": query.strip(), "k": k, "model_name": model_name.strip()}
    for name, default, lower, upper in (("temperature", 0.0, 0.0, 2.0), ("top_p", 1.0, 0.0, 1.0)):
        value = data.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be a finite number from {lower} to {upper}.")
        if name == "top_p" and value == 0:
            raise ValueError("top_p must be greater than zero.")
        parameters[name] = value
    max_tokens = data.get("max_tokens", 512)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be an integer from 1 to 4096.")
    parameters["max_tokens"] = max_tokens
    return parameters


def create_app(vectorstore=None):
    """Load the store once per worker, or accept a store for offline tests."""
    rag_api = Flask("Aurora Stay API")
    rag_api.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    store = load_vectorstore() if vectorstore is None else vectorstore
    rag_api.config["VECTORSTORE"] = store

    @rag_api.get("/")
    def home():
        return "Welcome to the Aurora Stay RAG API!"

    @rag_api.post("/v1/relevant_chunks")
    def relevant_chunks():
        try:
            parameters = request_parameters()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        chunks = retrieve(parameters["query"], k=parameters["k"], vectorstore=store)
        return jsonify({"query": parameters["query"], "k": parameters["k"], "relevant_chunks": chunks})

    def answer_response(include_chunks):
        try:
            parameters = request_parameters()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        try:
            answer, chunks = rag(**parameters, vectorstore=store)
        except Exception as error:
            # Provider exception messages can contain request/credential details.
            rag_api.logger.warning("Answer generation failed (%s).", type(error).__name__)
            return jsonify({"error": "Answer generation failed. Please retry or contact the property team."}), 502
        payload = {"answer": answer}
        if include_chunks:
            payload.update(parameters, relevant_chunks=chunks)
        return jsonify(payload)

    @rag_api.post("/v1/answer_with_relevant_chunks")
    def answer_with_relevant_chunks():
        return answer_response(include_chunks=True)

    @rag_api.post("/v1/answer")
    def answer():
        return answer_response(include_chunks=False)

    return rag_api


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=7860, debug=False)
