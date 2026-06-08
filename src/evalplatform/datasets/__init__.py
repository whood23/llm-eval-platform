"""Evalset as a first-class artifact (Phase 3).

Scaffolded (AI): ``loader`` (read/write JSONL eval items + candidates) and ``versioning``
(content-hash dataset versions + dataset cards). Hand-coded (YOU): ``sampling`` —
stratified sampling logic (the diversity/coverage *metric* lives in ``stats.diversity``).
"""
