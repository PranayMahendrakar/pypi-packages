"""offline-ml: detect the machine you are on and pick a model that will fit.

Two calls. The first looks at the machine, the second picks from the models you
are considering and explains the choice in plain language::

    import offline_ml

    print(offline_ml.detect().summary())
    print(offline_ml.recommend(models).summary())

Nothing here needs a network, a GPU, or a GPU library. Having no GPU is the
normal case, not an error: ``nvidia-smi`` missing or failing is expected, torch
is used only if it already happens to be installed, and the answer is then
simply ``device == "cpu"``.

Units: every ``*_gb`` number is a binary gigabyte (GiB, 1024**3 bytes), and
every frequency is in MHz.
"""
from ._gpu import BACKENDS, GPU
from ._hardware import Hardware, best_device, detect
from ._models import ModelSpec
from ._recommend import PREFERENCES, Recommendation, fits, recommend

__version__ = "0.1.0"

__all__ = [
    "BACKENDS",
    "GPU",
    "Hardware",
    "ModelSpec",
    "PREFERENCES",
    "Recommendation",
    "best_device",
    "detect",
    "fits",
    "recommend",
    "__version__",
]
