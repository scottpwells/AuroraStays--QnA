"""Evaluated notebook RAG functions; server credentials come from environment variables."""

import os
from langchain_openai import ChatOpenAI

# Combine keyword matching and FAISS, returning up to k distinct policy contexts.
from policy_retrieval import retrieve_policy_documents

def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    results = retrieve_policy_documents(vectorstore, query, k=k)
    return [
        f"Content:\n{doc.page_content}\n\nMetadata:\n" + str({
            "source": doc.metadata["source"],
            "sources": doc.metadata.get("sources", [doc.metadata["source"]]),
            "section": doc.metadata.get("heading", ""),
        })
        for doc in results
    ]

# Define the generation function that prompts the LLM with the retrieved context.
def generate(query, retrieved_chunks, model):
    rules = """
    - Use only the retrieved policy information to answer the question.
    - Match the requested property exactly. Preserve fee amounts, currencies,
      per-night/per-stay units, caps, conditions, and exceptions.
    - If the information needed to answer is present, give a direct, concise answer.
      Other properties' rows do not override the requested property's row.
    - Cite the source filename from the metadata of the chunk containing each
      fact used in the answer. Do not cite a different chunk about another property.
    - If the retrieved information is insufficient or conflicting, explain what
      is missing or conflicting and ask for verification with the current property
      profile or property team. Do not invent policy details or use outside knowledge.
    """
    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """
    return model.invoke(prompt)

# Combine retrieval and generation, returning answer text and source excerpts.
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
):
    if vectorstore is None:
        raise ValueError("Provide the initialized FAISS vectorstore.")

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"].strip(),
        timeout=60,
        max_retries=0,
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
