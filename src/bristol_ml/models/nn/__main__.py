"""``python -m bristol_ml.models.nn`` — delegate to the mlp module's CLI.

DESIGN §2.1.1 requires every module to run standalone; plan NFR-6 names
``python -m bristol_ml.models.nn.mlp --help`` as the Stage 10 entry
point.  ``python -m bristol_ml.models.nn`` is a convenience alias that
delegates to the same CLI, so the package and the module both resolve
the config schema.
"""

from __future__ import annotations

from bristol_ml.models.nn.mlp import _cli_main

if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
