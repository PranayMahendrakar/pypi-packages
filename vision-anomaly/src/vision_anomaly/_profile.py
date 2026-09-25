"""The fitted profile: what "normal" looked like, and how far something is from it.

Fitting reduces a set of known-good images to two numbers per feature - the
median and a robust spread - and scoring turns any other image into the same
feature vector and asks how many robust sigmas it sits away.

Why median and MAD rather than mean and standard deviation: a normal-set folder
almost always has one or two images in it that should not be there. A single
badly wrong image moves a mean and inflates a standard deviation enough to hide
every real anomaly behind it. The median and the median absolute deviation both
survive up to half the set being wrong, so a profile fitted on a slightly dirty
folder still detects.

The profile is plain data. It saves as JSON and loads as JSON: no pickle, so a
profile file can be read, diffed, checked into a repository and handed to a
different Python without executing anything.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ._features import (
    GROUP_NAMES,
    FeatureConfig,
    FeatureSpace,
    build_space,
)

#: Version tag written into every profile file. Refused on load if it differs,
#: rather than silently scoring against a layout that has since changed.
PROFILE_FORMAT = "vision-anomaly/profile/1"

#: Turns a median absolute deviation into something comparable with a standard
#: deviation, for data that is normal in the middle.
MAD_TO_SIGMA = 1.4826

#: Turns an interquartile range into the same units, for data that is normal in
#: the middle. The IQR is the second spread estimate taken of every feature; see
#: :func:`fit_profile` for why one is not enough.
IQR_TO_SIGMA = 1.0 / 1.349

#: Smallest spread any feature is allowed to have, in feature units. Every
#: feature this package computes is a 0.0 to 1.0 quantity - a fraction of
#: pixels, or a mean of a 0..1 plane - so one absolute floor is meaningful for
#: all of them. 0.002 is about half a grey level out of 255, or two pixels in a
#: thousand landing in a histogram bin: below that, a difference is measurement
#: noise, not news. Without this floor a feature that happened to be identical
#: across the fitted set - a histogram bin nothing ever falls in - would divide
#: by zero and make every later image infinitely anomalous.
NOISE_FLOOR = 0.002

#: Smallest spread a feature may have relative to its own level. A dozen images
#: never show all the nuisance variation a camera, a lamp and a hand placing a
#: part will produce, so a feature whose fitted values agree to better than 5%
#: of their own size is taken to agree to 5% and no better. This is the single
#: setting that trades false alarms against sensitivity: lower it and faint
#: departures start to register along with more noise.
RELATIVE_FLOOR = 0.05

#: Fitted images below this count get a "weak profile" warning. Under five
#: images the MAD of a feature is estimated from a handful of numbers and is as
#: likely to be too small as too large, which shows up as false alarms.
WEAK_PROFILE_IMAGES = 5

#: How far a feature may stray before it counts towards the score at all. A
#: held-out good image still moves every feature a little, and summing those
#: wobbles would give every clean image a floor of about one sigma and leave no
#: room to say "this one is identical". Only the distance past this band is
#: counted, which is what makes a copy of a fitted image score near zero.
INLIER_BAND = 3.0


def small_sample_factor(count: int) -> float:
    """Correction that stops a MAD from a handful of images reading too tight.

    The median absolute deviation of a small sample is biased low. This is the
    usual ``n / (n - 0.8)`` approximation to the exact correction: 1.19 at five
    images, 1.04 at twenty, 1.008 at a hundred. Leaving it out makes a profile
    fitted on six images call ordinary variation an anomaly.
    """
    count = int(count)
    if count <= 1:
        return 1.0
    return float(count) / (float(count) - 0.8)


@dataclass
class Deviation:
    """How one image's features stood against a profile.

    Attributes:
        z: signed distance of every feature, in robust sigmas.
        excess: how far past :data:`INLIER_BAND` each feature reached, zero for
            anything still inside it.
        score: the headline number - the largest of the six family scores.
        group_scores: each family's score: the quadratic mean of its features'
            excess.
        cell_scores: one number per grid cell, in row order.
    """

    z: np.ndarray
    excess: np.ndarray
    score: float
    group_scores: Dict[str, float]
    cell_scores: np.ndarray


@dataclass
class Profile:
    """Median and robust spread of every feature over the fitted set.

    Attributes:
        config: the feature layout this profile was fitted with.
        median: the median of every feature over the fitted images.
        scale: the robust spread of every feature, floored at
            :data:`NOISE_FLOOR`.
        n_images: how many images were fitted.
        sizes: the distinct source sizes seen, largest area first.
        notes: what happened while reading the fitted set.
        warnings: anything that makes this profile less trustworthy.
        fit_scores: the score each fitted image gets against this profile,
            which is the honest yardstick for reading any other score.
        value_scale: the divisor integer pixels were put on the 0..1 scale with
            while fitting, or ``None`` when the fitted set carried no integer
            image. Scoring reuses it so a later frame in a wider container is
            read the same way the profile was built, rather than guessed at
            again from its own contents.
    """

    config: FeatureConfig
    median: np.ndarray
    scale: np.ndarray
    n_images: int
    sizes: Tuple[Tuple[int, int], ...] = ()
    notes: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    fit_scores: Tuple[float, ...] = ()
    value_scale: Optional[float] = None
    _space: Optional[FeatureSpace] = field(default=None, repr=False, compare=False)

    @property
    def space(self) -> FeatureSpace:
        """The feature layout, built once and cached."""
        space = self._space
        if space is None:
            space = build_space(self.config)
            self._space = space
        return space

    @property
    def n_features(self) -> int:
        """How many numbers each image is reduced to."""
        return int(self.median.shape[0])

    @property
    def typical_fit_score(self) -> float:
        """The median score of the fitted images themselves.

        A test image scoring around this number looks exactly as normal as the
        set it was compared against.
        """
        if not self.fit_scores:
            return 0.0
        return float(np.median(np.asarray(self.fit_scores, dtype=np.float64)))

    @property
    def worst_fit_score(self) -> float:
        """The highest score any fitted image gets against this profile.

        Anything a user scores below this was no stranger than the least typical
        image they called normal.
        """
        if not self.fit_scores:
            return 0.0
        return float(np.max(np.asarray(self.fit_scores, dtype=np.float64)))

    def deviation(self, vector: np.ndarray) -> Deviation:
        """Measure one feature vector against this profile."""
        values = np.asarray(vector, dtype=np.float64).ravel()
        if values.shape[0] != self.n_features:
            raise ValueError(
                "feature vector has {0} numbers but the profile was fitted with "
                "{1}".format(values.shape[0], self.n_features)
            )
        z = (values - self.median) / self.scale
        excess = np.maximum(np.abs(z) - INLIER_BAND, 0.0)

        space = self.space
        group_scores: Dict[str, float] = {}
        for group in GROUP_NAMES:
            index = space.group_index(group)
            group_scores[group] = _quadratic_mean(excess[index]) if index.size else 0.0
        # The worst family wins, rather than the average of the six. Averaging
        # would divide a real colour fault by the five families that are fine
        # and hide it; every family is near zero for an image that belongs, so
        # the maximum is still near zero when nothing is wrong. Each family's
        # own number is a quadratic mean, so no single wild feature can raise it
        # on its own.
        score = max(group_scores.values()) if group_scores else 0.0

        cells = space.cells
        cell_scores = np.zeros(cells, dtype=np.float64)
        for cell in range(cells):
            index = space.cell_index(cell)
            if index.size:
                cell_scores[cell] = _quadratic_mean(excess[index])

        return Deviation(
            z=z,
            excess=excess,
            score=float(score),
            group_scores=group_scores,
            cell_scores=cell_scores,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the whole profile."""
        return {
            "format": PROFILE_FORMAT,
            "features": self.config.to_dict(),
            "n_images": int(self.n_images),
            "n_features": self.n_features,
            "inlier_band": float(INLIER_BAND),
            "noise_floor": float(NOISE_FLOOR),
            "median": [float(value) for value in self.median],
            "scale": [float(value) for value in self.scale],
            "sizes": [[int(w), int(h)] for w, h in self.sizes],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "fit_scores": [round(float(value), 6) for value in self.fit_scores],
            "value_scale": (
                None if self.value_scale is None else float(self.value_scale)
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        """Rebuild a profile from :meth:`to_dict` output.

        Raises:
            ValueError: the data is not a profile, was written by a different
                format version, or its arrays do not line up with its layout.
        """
        if not isinstance(data, dict):
            raise ValueError(
                "a profile must be a JSON object, got {0}".format(type(data).__name__)
            )
        found = data.get("format")
        if found != PROFILE_FORMAT:
            raise ValueError(
                "not a vision-anomaly profile: expected format {0!r}, found {1!r}".format(
                    PROFILE_FORMAT, found
                )
            )
        config = FeatureConfig.from_dict(dict(data.get("features") or {}))
        median = np.asarray(data.get("median") or [], dtype=np.float64)
        scale = np.asarray(data.get("scale") or [], dtype=np.float64)
        space = build_space(config)
        if median.shape != scale.shape:
            raise ValueError(
                "profile is damaged: {0} medians but {1} scales".format(
                    median.shape[0], scale.shape[0]
                )
            )
        if median.shape[0] != len(space):
            raise ValueError(
                "profile is damaged: {0} features stored, but its layout needs "
                "{1}".format(median.shape[0], len(space))
            )
        if not np.isfinite(median).all() or not np.isfinite(scale).all():
            raise ValueError("profile is damaged: it contains NaN or infinite values")
        if float(scale.min()) <= 0.0:
            raise ValueError("profile is damaged: it contains a zero or negative scale")
        sizes = tuple(
            (int(item[0]), int(item[1]))
            for item in (data.get("sizes") or [])
            if len(item) == 2
        )
        return cls(
            config=config,
            median=median,
            scale=scale,
            n_images=int(data.get("n_images", 0)),
            sizes=sizes,
            notes=tuple(str(note) for note in (data.get("notes") or [])),
            warnings=tuple(str(note) for note in (data.get("warnings") or [])),
            fit_scores=tuple(float(value) for value in (data.get("fit_scores") or [])),
            value_scale=(
                None
                if data.get("value_scale") is None
                else float(data["value_scale"])
            ),
            _space=space,
        )

    def save(self, path: Any) -> str:
        """Write the profile to ``path`` as JSON. Returns the path written."""
        target = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(target))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path: Any) -> "Profile":
        """Read a profile written by :meth:`save`.

        Raises:
            ValueError: the file is missing, is not JSON, or is not a profile.
        """
        target = os.fspath(path)
        if not os.path.exists(target):
            raise ValueError("no such profile file: {0}".format(target))
        if os.path.isdir(target):
            raise ValueError("{0} is a directory, not a profile file".format(target))
        try:
            with open(target, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "cannot read profile {0}: not valid JSON ({1})".format(target, exc.msg)
            ) from None
        except OSError as exc:
            raise ValueError(
                "cannot read profile {0}: {1}".format(target, exc.strerror or exc)
            ) from None
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError("cannot read profile {0}: {1}".format(target, exc)) from None

    def summary(self) -> str:
        """Plain text describing what this profile was fitted on."""
        lines = [
            "profile: {0} image{1}, {2} features, {3}x{3} grid on a {4}x{4} "
            "analysis image".format(
                self.n_images,
                "" if self.n_images == 1 else "s",
                self.n_features,
                self.config.grid,
                self.config.analysis_size,
            )
        ]
        if self.fit_scores:
            lines.append(
                "the fitted images themselves score {0:.2f} typical, {1:.2f} at "
                "worst".format(self.typical_fit_score, self.worst_fit_score)
            )
        if self.sizes:
            shown = ", ".join("{0}x{1}".format(w, h) for w, h in self.sizes[:4])
            more = "" if len(self.sizes) <= 4 else " and {0} more".format(
                len(self.sizes) - 4
            )
            lines.append("source sizes: {0}{1}".format(shown, more))
        for note in self.notes:
            lines.append("note: " + note)
        for warning in self.warnings:
            lines.append("warning: " + warning)
        return "\n".join(lines)


def _quadratic_mean(values: np.ndarray) -> float:
    """Root mean square, the aggregate used to pool features.

    Used for the features of a family and again for the features of a grid cell.
    It answers "how far, overall" without one wild feature running away with the
    answer, which a maximum would let it do, and without being pinned near zero
    by the quiet majority, which a plain mean would do.
    """
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def fit_profile(
    matrix: np.ndarray,
    config: FeatureConfig,
    *,
    sizes: Sequence[Tuple[int, int]] = (),
    notes: Sequence[str] = (),
    warnings: Sequence[str] = (),
    value_scale: Optional[float] = None,
) -> Profile:
    """Reduce a matrix of feature vectors to a profile.

    Args:
        matrix: one row per fitted image, one column per feature.
        config: the layout the rows were built with.
        sizes: the distinct source sizes the images came in.
        notes: anything worth telling the user about how they were read.
        warnings: anything that makes the profile less trustworthy.
        value_scale: the divisor integer pixels were read with, recorded so
            scoring can read later images the same way.

    Returns:
        The fitted profile, including each fitted image's own score against it.

    The spread of each feature is the largest of four numbers, and each one is
    there because leaving it out produced false alarms:

    * the **MAD**, scaled to a sigma and corrected for the sample size. The main
      estimate, and the one that survives a bad image in the normal set.
    * the **interquartile range**, scaled the same way. The MAD asks how far a
      typical image is from the middle, which collapses when a feature is
      two-valued rather than spread - a part that sits either side of a cell
      boundary, a lamp that is on or off. The IQR spans both clusters and keeps
      a quarter of the set's worth of protection at each end.
    * a **relative floor**, :data:`RELATIVE_FLOOR` of the feature's own level.
    * an **absolute floor**, :data:`NOISE_FLOOR`, which is what stops a feature
      that never moved at all from dividing by zero.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("fitting needs at least one row of features")
    space = build_space(config)
    if values.shape[1] != len(space):
        raise ValueError(
            "feature matrix has {0} columns but the layout needs {1}".format(
                values.shape[1], len(space)
            )
        )

    correction = small_sample_factor(values.shape[0])
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    spread = mad * MAD_TO_SIGMA * correction
    quartiles = np.percentile(values, [25.0, 75.0], axis=0)
    spread = np.maximum(spread, (quartiles[1] - quartiles[0]) * IQR_TO_SIGMA * correction)
    spread = np.maximum(spread, np.abs(median) * RELATIVE_FLOOR)
    scale = np.maximum(spread, NOISE_FLOOR)

    profile = Profile(
        config=config,
        median=median,
        scale=scale,
        n_images=int(values.shape[0]),
        sizes=tuple((int(w), int(h)) for w, h in sizes),
        notes=tuple(str(note) for note in notes),
        warnings=tuple(str(note) for note in warnings),
        value_scale=None if value_scale is None else float(value_scale),
        _space=space,
    )
    profile.fit_scores = tuple(
        profile.deviation(values[row]).score for row in range(values.shape[0])
    )
    return profile
