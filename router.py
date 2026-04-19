"""
Agentic RAG Router using LlamaIndex.

Given a query, the router selects one of two query engines:
  - Vector (Q&A): retrieves specific context from the document to answer questions.
  - Summary: summarizes the entire document.

Usage:
    export OPENAI_API_KEY=<your-key>
    python router.py
"""

import os

from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_router_query_engine(data_dir: str = DATA_DIR) -> RouterQueryEngine:
    """Load documents and build a router over Q&A and summarization query engines."""
    documents = SimpleDirectoryReader(data_dir).load_data()

    # Index for precise Q&A (vector similarity search)
    vector_index = VectorStoreIndex.from_documents(documents)

    # Index for full-document summarization
    summary_index = SummaryIndex.from_documents(documents)

    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # Wrap each engine in a tool with a description the router uses to choose
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for answering specific questions about the document. "
            "Use this for targeted Q&A queries."
        ),
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarizing the contents of the document. "
            "Use this when the query asks for a summary or overview."
        ),
    )

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=True,
    )

    return router


def main() -> None:
    router = build_router_query_engine()

    queries = [
        "What is the history of artificial intelligence?",
        "Can you give me a summary of the document?",
        "Who coined the term artificial intelligence?",
        "Summarize the key challenges facing AI today.",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("=" * 70)
        response = router.query(query)
        print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
