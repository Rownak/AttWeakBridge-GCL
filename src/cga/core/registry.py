"""Generic name-to-class registry used by every pluggable component category.

Pipelines, dataset adapters, embedders, samplers, and losses are each looked up by
a short string key (e.g. ``--dataset attack_cwe``, ``embedder: hf_mean_pool``). A
single generic ``Registry`` backs all of them so registration and lookup behave
identically everywhere, and adding a new dataset/embedder/etc. never requires
touching framework code.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from cga.core.errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object registry with decorator-based registration.

    Args:
        kind: Human-readable category name (e.g. "pipeline", "adapter"), used only
            to make error messages actionable.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers ``name -> obj`` on application.

        Raises:
            ConfigError: If ``name`` is already registered.
        """

        def decorator(obj: T) -> T:
            if name in self._items:
                raise ConfigError(
                    f"{self._kind} '{name}' is already registered "
                    f"(existing: {self._items[name]!r}, new: {obj!r})"
                )
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Look up a registered object by name.

        Raises:
            ConfigError: If ``name`` is not registered; message lists known names.
        """
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "(none registered)"
            raise ConfigError(
                f"Unknown {self._kind} '{name}'. Known {self._kind}s: {known}"
            ) from exc

    def names(self) -> list[str]:
        """Return all registered names, sorted."""
        return sorted(self._items)


pipelines: Registry = Registry("pipeline")
adapters: Registry = Registry("adapter")
embedders: Registry = Registry("embedder")
samplers: Registry = Registry("sampler")
losses: Registry = Registry("loss")
