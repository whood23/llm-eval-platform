"""The judge: run loop + reliability plumbing, and the hand-coded prompt/parse/swap logic.

Scaffolded (AI): ``runner`` (batching/concurrency), ``cache``, ``retry``, ``ratelimit``.
Hand-coded (YOU): ``pointwise`` and ``pairwise`` — the prompt/rubric, the output parser,
and the pairwise swap logic that presents each pair in BOTH orders. Those are left as
stubs that raise ``NotImplementedError`` until you implement them.
"""
