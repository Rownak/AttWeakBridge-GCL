"""``evaluate`` pipeline: comparative evaluation over embedding methods (target_architecture.md §5.5).

Importing this package registers the ``evaluate`` pipeline and every
built-in evaluator (``neighborhood``, ``cluster_quality``) as a side effect.
"""

from __future__ import annotations

from cga.pipelines.evaluate import evaluators  # noqa: F401  (registers built-in evaluators)
from cga.pipelines.evaluate import pipeline  # noqa: F401  (registers the pipeline)
