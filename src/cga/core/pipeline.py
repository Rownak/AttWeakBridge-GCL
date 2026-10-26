"""Pipeline base class and Step protocol shared by all four registered pipelines.

A ``Pipeline`` validates its config, runs its steps in declared order against a
shared ``RunContext``, writes the manifest, and marks the run ``success``/``failed``.
Per-item failures inside a step (e.g. one node's embedding) are the step's own
responsibility to catch and record — only exceptions that escape a step abort
the whole run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Protocol

from cga.core.run import RunContext


class Step(Protocol):
    """A single named unit of work within a pipeline."""

    name: str

    def run(self, ctx: RunContext) -> None:
        """Execute this step against the shared run context."""
        ...


class Pipeline(ABC):
    """Base class for registered pipelines (ingest, embed, training_data, gcl_train).

    Subclasses implement :meth:`build_steps` to return the ordered list of steps
    for a given resolved config; :meth:`run` handles the common lifecycle: run
    steps in order, write the manifest, mark success/failure.
    """

    name: str

    @abstractmethod
    def build_steps(self, ctx: RunContext) -> list[Step]:
        """Return this pipeline's steps in execution order for ``ctx``."""

    def run(self, ctx: RunContext) -> None:
        """Run all steps in order, then write the manifest with a final status.

        On any exception, the manifest is written with ``status: failed`` before
        the exception is re-raised, so the CLI can log and exit non-zero while
        the manifest still reflects what happened.
        """
        started = datetime.now(timezone.utc)
        try:
            for step in self.build_steps(ctx):
                ctx.logger.info("Starting step '%s'", step.name)
                step.run(ctx)
                ctx.logger.info("Finished step '%s'", step.name)
        except Exception:
            finished = datetime.now(timezone.utc)
            ctx.manifest.mark_failed(finished.isoformat(), (finished - started).total_seconds())
            ctx.manifest.write(ctx.run_dir)
            raise
        else:
            finished = datetime.now(timezone.utc)
            ctx.manifest.mark_success(finished.isoformat(), (finished - started).total_seconds())
            ctx.manifest.write(ctx.run_dir)
