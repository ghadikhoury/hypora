# S3.2 research

B uses an isolated MCP session with three read-only tools. Catalog queries use fixed public task
categories, never task text or dataset fields. Search results produce session-scoped opaque
references; only a reference from that session can fetch exact-revision metadata and a model card.
The single-agent baseline has the same tools; C retains its existing tools. Call, result, policy,
and model events and the run guard remain in force.

Hugging Face model cards are third-party, self-reported text. B sees them as untrusted evidence.
Returned records include a source URL, immutable revision, retrieval time, declared license,
gating state, stale-cache state, and truncation. Model-card claims are not independent benchmarks;
unavailable prices and hardware requirements remain unknown. Local retrieval only indexes cards
with a recognized open license and explicit ungated state. Unknown or gated records may be shown
with their limitations but are not added to the retrieval index.

The shared retrieval corpus splits cards at Markdown headings and caps sections. It stores
provenance with each passage and persists the lexical and dense index as JSON, never pickle.
Modes are off, TF-IDF, BM25, locally trained latent semantic analysis (TF-IDF plus truncated SVD),
and reciprocal-rank-fused BM25/latent semantic. The dense mode is not a pretrained or neural
embedding. Re-ranking remains deferred. Source/revision/license filters operate on returned
passages. A small index with insufficient rank falls back to its TF-IDF representation.

The 20 labeled source passages under tests/fixtures/retrieval are exact short descriptions from
the pinned scikit-learn package; the package version, URLs, license and expected passage references
are retained. They check source traceability and basic retrieval, but are narrow single-library API
questions and should not be presented as a representative evaluation of research quality. F1's
full comparison and error analysis belong to the planned evaluation step.

This design limits external egress but does not make untrusted text truthful. The model may still
misinterpret a source or accept an instruction in it; B cannot execute or delegate, and platform
code validates citation identity against tool results before accepting a strategy report.
