"""Logging setup: console handler plus an optional per-run file handler.

Configured once by the CLI at run start. The format string includes the pipeline
and step name so interleaved multi-step logs stay attributable without progress
bars (progress is reported every ``log_every`` items instead).
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    *,
    level: str = "INFO",
    run_dir: str | Path | None = None,
    to_file: bool = True,
) -> logging.Logger:
    """Configure and return the root ``cga`` logger.

    Args:
        level: Standard logging level name (e.g. "INFO", "DEBUG").
        run_dir: If given and ``to_file`` is true, logs are also written to
            ``{run_dir}/run.log``.
        to_file: Whether to attach a file handler when ``run_dir`` is provided.

    Returns:
        The configured ``cga`` logger.
    """
    logger = logging.getLogger("cga")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if to_file and run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``cga`` namespace (e.g. ``cga.pipelines.ingest``)."""
    return logging.getLogger(f"cga.{name}")
