"""Dataset adapter registry.

Importing this package registers every built-in adapter (target_architecture.md
§4.1) as a side effect, so ``registry.adapters.get(dataset.adapter)`` works after
a plain ``import cga.adapters``. Adding a dataset means adding one adapter
package here and importing it below — no other file changes.
"""

from __future__ import annotations

from cga.adapters import attack_cwe, ncit_doid  # noqa: F401
