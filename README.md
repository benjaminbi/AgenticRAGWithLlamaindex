# AgenticRAGWithLlamaindex

Agentic RAG with LlamaIndex – a **router agent** that selects the best query engine for a given query over a single document.

## How it works

Given a natural-language query, the router picks one of two query engines:

| Engine | Index | Best for |
|--------|-------|----------|
| **Q&A** | `VectorStoreIndex` (vector similarity search) | Targeted, specific questions |
| **Summarization** | `SummaryIndex` (tree summarize) | High-level summaries or overviews |

The routing decision is made by an LLM (`LLMSingleSelector`) that reads the tool descriptions and picks the most appropriate engine automatically.

```
Query ──► LLMSingleSelector ──► Vector Q&A engine   ──► Answer
                           └──► Summary engine       ──► Summary
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=<your-openai-api-key>
```

## Usage

Run the demo script against the sample document in `data/`:

```bash
python router.py
```

The script sends four example queries—two Q&A and two summarization—and prints the selected engine and response for each.

### Use in your own code

```python
from router import build_router_query_engine

router = build_router_query_engine()          # loads docs from ./data/
response = router.query("Who invented deep learning?")
print(response)
```

Replace the files in `data/` with any documents you want to query.

## Files

| File | Description |
|------|-------------|
| `router.py` | Core router implementation and demo |
| `data/sample.txt` | Sample document (AI overview) used for the demo |
| `requirements.txt` | Python dependencies |
