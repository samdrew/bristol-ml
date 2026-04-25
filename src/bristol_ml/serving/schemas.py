"""Pydantic v2 request / response schemas for the serving layer (Stage 12).

The full implementation lands at Stage 12 T6.  At T1 (this file) only
a placeholder docstring is in place so the package import surface is
complete; T6 adds the concrete :class:`PredictRequest` and
:class:`PredictResponse` models per ``docs/plans/active/12-serving.md`` §5.
"""

from __future__ import annotations

# Stage 12 T6 will populate this module with PredictRequest /
# PredictResponse.  The shape is fully specified in the plan §5; the
# T1 scaffold deliberately keeps the file empty of class definitions
# so ``import bristol_ml.serving.schemas`` works without pulling
# pydantic into the import graph until T6 actually adds the models
# (pydantic is already a runtime dep so this is cosmetic — but the
# discipline matches the project's lazy-import convention for heavy
# deps).
