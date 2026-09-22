"""ml-pipeline-kit: a preprocess, predict, validate and log pipeline in a few lines.

Every step is timed, its row counts recorded, and its failure reported with the
step's name instead of a bare traceback, without you writing any of that::

    from ml_pipeline_kit import Pipeline

    pipe = Pipeline("scoring").preprocess(clean).predict(score)
    result = pipe.run(frame)
    print(result.summary())

Inside a service, call the pipeline instead of running it and you get the output
value alone, with a clear error if a check stopped it::

    prediction = pipe(request_frame)

Checks live beside the steps and run in order with them::

    pipe = (Pipeline("scoring")
            .expect_schema({"age": "int", "city": "str"})
            .expect_range("age", 18, 100)
            .predict(score)
            .validate(lambda df: df["score"].notna().all(), name="scored"))
"""
from ._checks import dtype_matches
from ._errors import PipelineError, StepError, ValidationError
from ._pipeline import ON_ERROR, SEVERITIES, Pipeline, run
from ._result import Result, StepRun

__version__ = "0.1.0"

__all__ = [
    "ON_ERROR",
    "Pipeline",
    "PipelineError",
    "Result",
    "SEVERITIES",
    "StepError",
    "StepRun",
    "ValidationError",
    "dtype_matches",
    "run",
    "__version__",
]
