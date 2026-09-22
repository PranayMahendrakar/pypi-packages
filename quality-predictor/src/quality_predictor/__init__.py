"""Predict product quality from manufacturing parameters, before final inspection.

    import pandas as pd
    from quality_predictor import predict_quality

    result = predict_quality(df, "quality")
    print(result.summary())

The summary carries the held-out scores, the parameters that drive the outcome,
and the setting windows that go with good parts. For more control, use
:class:`QualityModel` directly: ``.fit()``, ``.predict()``, ``.evaluate()``,
``.feature_importance``, ``.explain(row)``, ``.optimal_ranges()``, ``.save()``.
"""

from ._api import predict_quality
from ._model import QualityModel
from ._result import Explanation, QualityResult

__version__ = "0.1.0"

__all__ = [
    "predict_quality",
    "QualityModel",
    "QualityResult",
    "Explanation",
    "__version__",
]
