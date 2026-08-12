# Architecture

## Product boundary

The system answers questions over a governed document collection. It is a retrieval
system with a bounded agent control layer, not an unrestricted chatbot.

```text
versioned document corpus
  -> contract validation
  -> deterministic stable chunks
  -> access-scope filter
  -> BM25 candidates + TF-IDF candidates
  -> Reciprocal Rank Fusion
  -> bounded route
  -> extractive evidence + validated citations or abstention
```

## Implemented smoke profile

- JSONL document contract with source URI, language, access scope, and provenance.
- Deterministic sentence-group chunk IDs.
- Vietnamese-English normalization.
- Access-scoped BM25 and sparse TF-IDF retrieval.
- RRF with component score and rank preservation.
- Lookup, comparison, and abstention routes.
- Extractive fallback with retrieved-chunk citations.
- Evaluation for Recall@k, MRR, abstention, and citation integrity.

No LLM is required for the core path. `allow_llm=false` is deliberate until an
adapter has bounded inputs, schema validation, timeout, cost limits, and the same
extractive fallback.

## Trust boundaries

Access scope is applied before candidates are scored. Answer citations are created
only from the returned authorized chunks. A production implementation must repeat
scope enforcement in the catalog, cache, index, answer, and logging layers.

## Development adapters

1. Dense multilingual encoder behind the existing retriever contract.
2. Cross-encoder reranking over bounded top-k.
3. Versioned persistent catalog and immutable serving release.
4. LLM answer adapter constrained to retrieved citation IDs.
5. API, observability, prompt-injection tests, and tenant isolation.
