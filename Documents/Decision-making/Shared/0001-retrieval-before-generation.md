# ADR-0001: Retrieval and authorization precede generation

## Status

Accepted.

## Decision

The initial agent uses access-scoped BM25 and TF-IDF retrieval, RRF, extractive
answers, citation validation, and abstention. An LLM is optional and disabled.

## Rationale

This makes retrieval quality, access control, citation integrity, and fallback
testable without model availability or API credentials. Future generators must
consume only retrieved chunk IDs and cannot change source identity or permissions.
