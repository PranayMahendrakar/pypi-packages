"""Turn a stage tree into plain-language advice.

Every rule here is a heuristic over the measured numbers, and each one names the evidence
it fired on, so a reader can disagree with it. The list is never empty: when nothing
stands out, the first line still says which stage is slowest and by how much.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import ProfileReport, Stage

#: Label fragments that mark a stage as data preparation rather than the model itself.
PREPROCESS_WORDS = (
    "preprocess",
    "pre-process",
    "pre_process",
    "prep",
    "tokeni",
    "resize",
    "rescale",
    "load",
    "read",
    "fetch",
    "transform",
    "feature",
    "normali",
    "augment",
    "encode",
    "parse",
    "clean",
    "scale",
    "input",
    # Moving the bytes is data-path work, not the model: a host-to-device copy or a
    # network fetch belongs here even when the label names the device.
    "upload",
    "copy",
    "transfer",
    "to_device",
    "h2d",
)

#: Label fragments that mark a stage as the model call. Unambiguous: any label containing
#: one of these is about the model, wherever in the label it appears.
MODEL_WORDS = (
    "model",
    "infer",
    "predict",
    "forward",
    "session",
    "onnx",
    "engine",
    "llm",
    "generate",
)

#: Weaker model hints, checked last and only on whole words. As bare substrings these are
#: wrong more often than right: "gpu" is in ``gpu_upload`` and "net" is in
#: ``network fetch``, both of which are data path, not model. Matched against a whole
#: token (trailing digits stripped) they still catch ``gpu`` and ``resnet50``, and they
#: only get a turn after the preprocessing and postprocessing words have had theirs.
WEAK_MODEL_WORDS = (
    "gpu",
    "net",
)

_TOKENS = re.compile(r"[^0-9a-z]+")

#: Label fragments that mark a stage as postprocessing.
#: Fragments that settle an otherwise ambiguous ``decode``: decoding an image is
#: input work, decoding tokens is output work.
DECODE_INPUT_WORDS = ("image", "jpeg", "jpg", "png", "video", "frame", "audio", "wav", "input")
DECODE_OUTPUT_WORDS = ("token", "text", "label", "output", "logit", "caption", "answer")

POSTPROCESS_WORDS = (
    "post",
    "detokeni",
    "nms",
    "softmax",
    "argmax",
    "format",
    "render",
    "serialize",
    "serialise",
    "output",
    "response",
)


def _matches(label: str, words: Sequence[str]) -> bool:
    lowered = label.lower()
    return any(word in lowered for word in words)


def _matches_word(label: str, words: Sequence[str]) -> bool:
    """Like :func:`_matches`, but only against whole tokens of the label.

    ``gpu_upload`` splits into ``gpu`` and ``upload``, so "gpu" matches it, while
    ``network`` does not match "net" - the point of the weaker tier. A trailing digit run
    is stripped first so that ``resnet50`` still reads as a net.
    """
    for token in _TOKENS.split(label.lower()):
        stem = token.rstrip("0123456789")
        if not stem:
            continue
        if any(stem == word or stem.endswith(word) for word in words):
            return True
    return False


def classify(label: str) -> str:
    """Guess what kind of work a label describes: ``preprocess``, ``model``,
    ``postprocess`` or ``other``.

    The unambiguous model words run first, so a stage called ``model input`` counts as the
    model. :data:`WEAK_MODEL_WORDS` run last, on whole tokens only, so a data-path stage
    named after the device or the wire - ``gpu_upload``, ``network fetch`` - is classified
    by the work it does rather than by the hardware it names.

    A label that matches nothing is ``other``; rules that need a class simply do not fire
    for it, and the always-on bottleneck line still reports the stage. This is a guess
    from a name, not a measurement: treat a surprising answer as the heuristic's fault.
    """
    if _matches(label, MODEL_WORDS):
        return "model"
    if _matches(label, POSTPROCESS_WORDS):
        return "postprocess"
    if "decode" in label.lower():
        # Ambiguous on its own: decoding a JPEG is input work, decoding tokens is
        # output work. Let the rest of the label decide, and stay out of the way
        # when it says nothing either way rather than guessing "preprocess" and
        # handing an output-heavy pipeline the exact opposite diagnosis.
        lowered = label.lower()
        if any(word in lowered for word in DECODE_INPUT_WORDS):
            return "preprocess"
        if any(word in lowered for word in DECODE_OUTPUT_WORDS):
            return "postprocess"
        return "other"
    if _matches(label, PREPROCESS_WORDS):
        return "preprocess"
    if _matches_word(label, WEAK_MODEL_WORDS):
        return "model"
    return "other"


def _self_ms_of_kind(stages: "Sequence[Stage]", kind: str) -> float:
    return sum(s.self_ms for s in stages if classify(s.label) == kind)


def build_suggestions(report: "ProfileReport") -> List[str]:
    """The advice shown by :attr:`ProfileReport.suggestions`, most useful line first."""
    stages = list(report.stages)
    if not stages:
        return [
            "No stages were recorded, so there is nothing to profile yet. Wrap the code"
            " you care about in 'with profiler.stage(\"name\"):', decorate a function with"
            " @profiler.profile, or pass a list of (label, function) steps to"
            " profile_pipeline()."
        ]

    out: List[str] = []
    total = report.total_ms
    out.append(_bottleneck_line(report))
    out.extend(_preprocessing_rule(stages, total))
    out.extend(_cold_cache_rule(stages, report))
    out.extend(_batching_rule(stages, report))
    out.extend(_overhead_rule(report))
    out.extend(_error_rule(stages))
    return out


def _bottleneck_line(report: "ProfileReport") -> str:
    top = report.bottleneck
    if top is None:  # pragma: no cover - the caller only calls this with stages
        return "Nothing was recorded."
    return (
        f"Slowest step: {top.path or top.label} spends {top.self_ms:.2f} ms in its own"
        f" code, {top.self_pct_of_total:.0f}% of the {report.total_ms:.2f} ms run, over"
        f" {top.calls} call(s). Start there."
    )


def _preprocessing_rule(stages: "Sequence[Stage]", total: float) -> List[str]:
    if total <= 0:
        return []
    prep = _self_ms_of_kind(stages, "preprocess")
    model = _self_ms_of_kind(stages, "model")
    share = 100.0 * prep / total
    # Half the run, not a third: a 40/35/25 split across preprocess, model and
    # postprocess is a healthy pipeline, and calling it out taught users to ignore
    # the advice. The always-on bottleneck line still names the slowest stage.
    if prep <= 0 or share < 50.0 or prep <= model:
        # Below a third of the run, or smaller than the model, preprocessing is not the
        # thing to fix first; the bottleneck line already says what is.
        return []
    line = (
        f"Preprocessing is {share:.0f}% of the run ({prep:.2f} ms): the data path, not the"
        " model, is where the time goes."
    )
    if model > 0:
        line += f" The model itself accounts for {100.0 * model / total:.0f}%."
    line += (
        " Cache decoded inputs, move resizing and tokenizing into the loader, or prepare"
        " the next batch on a worker thread while the current one runs."
    )
    return [line]


def _cold_cache_rule(stages: "Sequence[Stage]", report: "ProfileReport") -> List[str]:
    floor = max(report.overhead_ms_per_stage * 4.0, 0.05)
    out: List[str] = []
    for stage in stages:
        if stage.calls < 3 or stage.median_ms <= 0:
            continue
        if stage.p95_ms < 2.0 * stage.median_ms or stage.p95_ms - stage.median_ms < floor:
            continue
        if stage.pct_of_total < 5.0:
            continue
        line = (
            f"{stage.path or stage.label} has a p95 of {stage.p95_ms:.2f} ms against a"
            f" median of {stage.median_ms:.2f} ms, so a few calls are far slower than the"
            " rest."
        )
        if stage.first_ms >= stage.max_ms - 1e-9:
            line += (
                f" The slowest call is the first one ({stage.first_ms:.2f} ms): that is a"
                " cold cache, a lazily built model or a first-touch allocation. Run a"
                " warmup pass before measuring, and warm it at startup in production."
            )
        else:
            line += (
                " The spread is not just the first call, so look for an input-dependent"
                " path, a garbage collection pause, or contention with another process."
            )
        out.append(line)
    return out


def _batching_rule(stages: "Sequence[Stage]", report: "ProfileReport") -> List[str]:
    repeats = max(report.repeats, 1)
    out: List[str] = []
    for stage in stages:
        per_repeat = stage.calls / repeats
        if per_repeat < 8 or stage.mean_ms >= 5.0:
            continue
        if stage.pct_of_total < 10.0:
            continue
        # With one repeat the quotient IS the total, and printing it as a rate reads as
        # if each pass did that much work. Only say "per repeat" when there were several.
        if repeats > 1:
            counted = f"{stage.calls} time(s) ({per_repeat:.0f} per repeat)"
        else:
            counted = f"{stage.calls} time(s) in total"
        line = (
            f"{stage.path or stage.label} ran {counted}"
            f" at {stage.mean_ms:.3f} ms each, which is"
            f" {stage.pct_of_total:.0f}% of the run. That is per-item work: collect the"
            " items and do one batched call instead, so the fixed cost per call is paid"
            " once rather than per item."
        )
        if report.overhead_ms_per_stage > 0 and stage.mean_ms < report.overhead_ms_per_stage * 10:
            line += (
                f" Each call is also close to the {report.overhead_ms_per_stage:.4f} ms"
                " timing floor, so its own measurement is rough."
            )
        out.append(line)
    return out


def _overhead_rule(report: "ProfileReport") -> List[str]:
    total = report.total_ms
    overhead = report.overhead_total_ms
    if total <= 0 or overhead < 0.05 * total:
        return []
    return [
        f"Timing overhead is about {overhead:.3f} ms of the {total:.2f} ms measured"
        f" ({100.0 * overhead / total:.0f}%), because {report.stage_calls} stage calls"
        " were recorded. Stages this small cannot be timed accurately one at a time:"
        " wrap a batch of them in a single stage and divide."
    ]


def _error_rule(stages: "Sequence[Stage]") -> List[str]:
    out: List[str] = []
    for stage in stages:
        if not stage.errors:
            continue
        out.append(
            f"{stage.path or stage.label} raised on {stage.errors} of its {stage.calls}"
            " call(s). The elapsed time is still counted, but these numbers describe a"
            " run that failed, not a healthy one."
        )
    return out


__all__ = [
    "build_suggestions",
    "classify",
    "PREPROCESS_WORDS",
    "MODEL_WORDS",
    "WEAK_MODEL_WORDS",
    "POSTPROCESS_WORDS",
]
