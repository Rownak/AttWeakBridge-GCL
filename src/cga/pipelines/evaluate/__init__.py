"""``evaluate`` pipeline: neighborhood-preservation analysis (target_architecture.md §5.5a).

Phase 1.8 scope only: the label-free ``neighborhood`` step and optional
``visualize`` step. The alignment-ranking ``rank``/``diagnose`` steps (§5.5,
Phase 1.9) are not implemented here.
"""

from __future__ import annotations

from cga.pipelines.evaluate import pipeline  # noqa: F401  (registers the pipeline)
