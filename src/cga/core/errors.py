"""Typed exceptions raised across the cga package.

Using dedicated exception types (rather than bare ``ValueError``/``RuntimeError``)
lets the CLI catch failures by category, log them with the right context, and mark
the run manifest ``status: failed`` without guessing at intent from a message string.
"""


class CgaError(Exception):
    """Base class for all cga-raised exceptions."""


class ConfigError(CgaError):
    """Raised when configuration is missing, malformed, or fails validation."""


class MissingInputError(CgaError):
    """Raised when a required input file or upstream artifact cannot be found."""


class ArtifactValidationError(CgaError):
    """Raised when an artifact fails schema, hash, or referential-integrity checks."""
