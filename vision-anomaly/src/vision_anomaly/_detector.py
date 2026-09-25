"""The public surface: :class:`Detector` and :func:`detect_anomalies`.

Fit on images you are happy with, then ask about anything else. There is no
training, no labelled defect, no download and no model file: the "profile" is
the median and robust spread of a few hundred hand-built numbers, which is why
fitting on eight photographs takes about as long as opening them.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ._features import (
    DEFAULT_GRID,
    FeatureConfig,
    FeatureSpace,
    build_space,
    extract,
)
from ._loading import ANALYSIS_SIZE, LoadedImage, iter_images, load_image
from ._profile import WEAK_PROFILE_IMAGES, Profile, fit_profile
from ._result import (
    DEFAULT_REASONS,
    DEFAULT_REGIONS,
    AnomalyResult,
    BatchReport,
    build_result,
)

LOGGER = logging.getLogger(__name__)

#: Default sensitivity: how many robust sigmas outside normal an image must sit
#: before it is called anomalous. Lower catches more and cries wolf more.
DEFAULT_SENSITIVITY = 3.0

#: The message :meth:`Detector.score` and friends raise before anything is
#: fitted. Spelled out once so every entry point says the same thing.
NOT_FITTED = (
    "this Detector has not been fitted yet: call fit(images) with your "
    "known-good examples first, or Detector.load(path) to read a saved profile"
)


class Detector:
    """Learns what normal looks like from good images, then flags the rest.

    Args:
        sensitivity: how many robust sigmas outside normal an image must sit
            before :attr:`AnomalyResult.anomalous` is true. The default of 3.0
            suits a normal set that genuinely covers the variation you accept.
        grid: cells on a side of the coarse grid the regions are reported on.
            A finer grid localises better and tolerates less camera movement.
        analysis_size: edge of the square grid images are resampled onto. Must
            divide by ``grid``.
        max_regions: how many departing cells a result lists.
        max_reasons: how many feature families a result names.

    Example:
        >>> import numpy as np
        >>> from vision_anomaly import Detector
        >>> good = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(6)]
        >>> detector = Detector().fit(good)
        >>> round(detector.score(good[0]), 3)
        0.0
        >>> detector.predict(np.zeros((64, 64, 3), dtype=np.uint8)).anomalous
        True
    """

    def __init__(
        self,
        sensitivity: float = DEFAULT_SENSITIVITY,
        *,
        grid: int = DEFAULT_GRID,
        analysis_size: int = ANALYSIS_SIZE,
        max_regions: int = DEFAULT_REGIONS,
        max_reasons: int = DEFAULT_REASONS,
    ) -> None:
        sensitivity = float(sensitivity)
        if not sensitivity > 0.0 or not np.isfinite(sensitivity):
            raise ValueError(
                "sensitivity must be a positive number of robust sigmas, got "
                "{0!r}".format(sensitivity)
            )
        self.sensitivity = sensitivity
        self.max_regions = int(max_regions)
        self.max_reasons = int(max_reasons)
        self._config = FeatureConfig(
            analysis_size=int(analysis_size), grid=int(grid)
        ).validate()
        self._space: FeatureSpace = build_space(self._config)
        self._profile: Optional[Profile] = None

    # -- state -------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """Whether a profile is loaded or fitted."""
        return self._profile is not None

    @property
    def profile(self) -> Profile:
        """The fitted profile.

        Raises:
            RuntimeError: nothing has been fitted or loaded yet.
        """
        if self._profile is None:
            raise RuntimeError(NOT_FITTED)
        return self._profile

    @property
    def config(self) -> FeatureConfig:
        """The feature layout this detector measures with."""
        return self._config

    def __repr__(self) -> str:
        if self._profile is None:
            return "Detector(sensitivity={0:g}, unfitted)".format(self.sensitivity)
        return "Detector(sensitivity={0:g}, fitted on {1} images, {2} features)".format(
            self.sensitivity, self._profile.n_images, self._profile.n_features
        )

    # -- fitting -----------------------------------------------------------

    def fit(self, images: Any) -> "Detector":
        """Learn what normal looks like from known-good images.

        Args:
            images: a sequence of paths, PIL images or numpy arrays; a stacked
                ``NxHxWx3`` array; or the path of a directory of images. Every
                one of them is taken to be an example of normal.

        Returns:
            This detector, so ``Detector().fit(good)`` reads as one expression.

        Raises:
            ValueError: fewer than two usable images, or one of them could not
                be read. The message always names the file that failed.

        Warns:
            UserWarning: fewer than :data:`vision_anomaly._profile.WEAK_PROFILE_IMAGES`
                images. The profile still works, but the spread of each feature
                is estimated from a handful of numbers, so ordinary variation
                can read as an anomaly.
        """
        items = _as_image_list(images)
        if not items:
            raise ValueError(
                "fit needs known-good images to learn from, but none were given"
            )
        if len(items) == 1:
            _reject_missing_path(items[0])
            raise ValueError(
                "fit needs at least 2 known-good images, got 1: a profile built "
                "from a single image has no notion of how much variation is "
                "normal, so every other image would look anomalous. Give it the "
                "spread of good images you actually accept - 10 or more is "
                "comfortable"
            )

        loaded = [self._load(item) for item in items]
        value_scale, scale_warning = _resolve_value_scale(loaded)
        if scale_warning is not None:
            # The set disagreed about how deep its own pixels are. Re-read every
            # image on the one scale that won, so the profile is at least built
            # from one picture of normal rather than two.
            loaded = [self._load(item, value_scale=value_scale) for item in items]
        matrix = np.vstack([extract(image, self._space) for image in loaded])
        sizes = _distinct_sizes(loaded)
        notes = _fit_notes(loaded, sizes, self._config.analysis_size)
        warns: List[str] = []
        if scale_warning is not None:
            warns.append(scale_warning)
            warnings.warn(scale_warning, UserWarning, stacklevel=2)
        if len(items) < WEAK_PROFILE_IMAGES:
            message = (
                "this profile was fitted on only {0} images; under {1} the "
                "spread of each feature is estimated from a handful of numbers, "
                "so ordinary variation can read as an anomaly. It will work, but "
                "treat borderline scores with suspicion".format(
                    len(items), WEAK_PROFILE_IMAGES
                )
            )
            warns.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        self._profile = fit_profile(
            matrix,
            self._config,
            sizes=sizes,
            notes=notes,
            warnings=warns,
            value_scale=value_scale,
        )
        LOGGER.debug(
            "fitted on %d images, %d features, typical fit score %.3f",
            self._profile.n_images,
            self._profile.n_features,
            self._profile.typical_fit_score,
        )
        return self

    # -- scoring -----------------------------------------------------------

    def score(self, image: Any) -> float:
        """How far one image sits from normal, in robust sigmas.

        Zero means every feature landed inside the band the fitted set defines.
        An identical copy of a fitted image scores near zero - usually exactly
        zero, and never above :attr:`Profile.worst_fit_score`, which is what a
        score should be read against.

        Args:
            image: a path, a PIL image or a numpy array.

        Returns:
            The distance, never negative.

        Raises:
            RuntimeError: nothing has been fitted or loaded yet.
            ValueError: the image could not be read; the message names it.
        """
        profile = self.profile
        loaded = self._load(image, value_scale=profile.value_scale)
        return float(profile.deviation(extract(loaded, self._space)).score)

    def predict(self, image: Any) -> AnomalyResult:
        """Score one image and explain the answer.

        Args:
            image: a path, a PIL image or a numpy array.

        Returns:
            The verdict, with the regions and reasons behind it.

        Raises:
            RuntimeError: nothing has been fitted or loaded yet.
            ValueError: the image could not be read; the message names it.
        """
        profile = self.profile
        loaded = self._load(image, value_scale=profile.value_scale)
        deviation = profile.deviation(extract(loaded, self._space))
        return build_result(
            source=loaded.source,
            deviation=deviation,
            space=self._space,
            threshold=self.sensitivity,
            fit_typical=profile.typical_fit_score,
            fit_worst=profile.worst_fit_score,
            n_fitted=profile.n_images,
            notes=_predict_notes(loaded, profile, self._config.analysis_size),
            warnings=profile.warnings,
            size=loaded.size,
            channels=loaded.channels,
            max_regions=self.max_regions,
            max_reasons=self.max_reasons,
        )

    def predict_batch(self, images: Any) -> BatchReport:
        """Score every image in one go and collect the answers.

        Args:
            images: a sequence of paths, PIL images or numpy arrays; a stacked
                ``NxHxWx3`` array; or the path of a directory of images.

        Returns:
            One report, holding a result per image in the order given.

        Raises:
            RuntimeError: nothing has been fitted or loaded yet.
            ValueError: an image could not be read; the message names it.
        """
        profile = self.profile
        items = _as_image_list(images)
        results = [self.predict(item) for item in items]
        return BatchReport(
            results=results,
            threshold=self.sensitivity,
            n_fitted=profile.n_images,
            fit_typical=profile.typical_fit_score,
            fit_worst=profile.worst_fit_score,
            warnings=list(profile.warnings),
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: Any) -> str:
        """Write the fitted profile to ``path`` as JSON.

        The file is plain JSON, not a pickle: it can be read, diffed and checked
        into a repository, and loading it executes nothing.

        Args:
            path: where to write. Parent directories are created.

        Returns:
            The path written.

        Raises:
            RuntimeError: nothing has been fitted or loaded yet.
        """
        return self.profile.save(path)

    @classmethod
    def load(
        cls, path: Any, *, sensitivity: Optional[float] = None, **kwargs: Any
    ) -> "Detector":
        """Read a profile written by :meth:`save` into a ready detector.

        Args:
            path: the profile file.
            sensitivity: override the threshold; the default is used otherwise.
            **kwargs: passed to the constructor, for ``max_regions`` and friends.

        Returns:
            A fitted detector, measuring with the layout the profile was fitted
            with rather than the current defaults.

        Raises:
            ValueError: the file is missing, is not JSON, or is not a profile.
        """
        profile = Profile.load(path)
        detector = cls(
            sensitivity=DEFAULT_SENSITIVITY if sensitivity is None else sensitivity,
            grid=profile.config.grid,
            analysis_size=profile.config.analysis_size,
            **kwargs,
        )
        detector._config = profile.config
        detector._space = profile.space
        detector._profile = profile
        return detector

    def summary(self) -> str:
        """Plain text describing what this detector was fitted on."""
        if self._profile is None:
            return "Detector: not fitted yet (sensitivity {0:.2f})".format(
                self.sensitivity
            )
        return "sensitivity {0:.2f}\n{1}".format(
            self.sensitivity, self._profile.summary()
        )

    # -- internals ---------------------------------------------------------

    def _load(
        self, image: Any, value_scale: Optional[float] = None
    ) -> LoadedImage:
        return load_image(
            image, size=self._config.analysis_size, value_scale=value_scale
        )


def detect_anomalies(
    good_images: Any,
    test_images: Any,
    *,
    sensitivity: float = DEFAULT_SENSITIVITY,
    **kwargs: Any,
) -> BatchReport:
    """Fit on the good images and check the test images, in one call.

    Args:
        good_images: known-good examples - paths, PIL images, arrays, a stacked
            array, or a directory path.
        test_images: the images to check, in any of the same forms.
        sensitivity: how many robust sigmas outside normal counts as anomalous.
        **kwargs: passed to :class:`Detector`, for ``grid`` and friends.

    Returns:
        One :class:`~vision_anomaly._result.BatchReport` covering every test
        image.

    Example:
        >>> import numpy as np
        >>> from vision_anomaly import detect_anomalies
        >>> good = [np.full((48, 48, 3), 100, dtype=np.uint8) for _ in range(6)]
        >>> report = detect_anomalies(good, [np.zeros((48, 48, 3), np.uint8)])
        >>> report.n_anomalous
        1
    """
    detector = Detector(sensitivity=sensitivity, **kwargs).fit(good_images)
    return detector.predict_batch(test_images)


def _as_image_list(images: Any) -> List[Any]:
    """Normalise every accepted spelling of "some images" into a flat list.

    A directory path becomes the images in it; a stacked ``NxHxWx3`` array
    becomes its frames; one image becomes a list of one, so the caller gets the
    "fit needs at least 2" message rather than something about iteration.
    """
    if images is None:
        return []
    if isinstance(images, (str, bytes, os.PathLike)):
        path = os.fspath(images)
        if isinstance(path, bytes):
            path = path.decode("utf-8", errors="replace")
        if os.path.isdir(path):
            found = iter_images(path)
            if not found:
                raise ValueError("no image files found in directory: {0}".format(path))
            return list(found)
        return [path]
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, np.ndarray):
        if images.ndim == 4:
            return [images[index] for index in range(images.shape[0])]
        return [images]
    if isinstance(images, LoadedImage):
        return [images]
    if isinstance(images, Iterable):
        return list(images)
    raise TypeError(
        "images must be a sequence of paths, PIL images or arrays, a stacked "
        "NxHxWx3 array, or a directory path; got " + type(images).__name__
    )


def _reject_missing_path(only: Any) -> None:
    """Say the path is missing before complaining that there is only one of it.

    ``fit("typo_dir")`` is a mistyped folder far more often than it is a genuine
    one-image fit set, and being told to supply more images when the real
    problem is a name that does not exist sends the reader looking in the wrong
    place entirely.
    """
    if not isinstance(only, (str, bytes, os.PathLike)):
        return
    path = os.fspath(only)
    if isinstance(path, bytes):
        path = path.decode("utf-8", errors="replace")
    if not os.path.exists(path):
        raise ValueError("no such image file: {0}".format(path))


def _resolve_value_scale(
    loaded: Sequence[LoadedImage],
) -> Tuple[Optional[float], Optional[str]]:
    """The one integer scale the fitted set is read on, and any complaint.

    Every integer image is put on the 0..1 analysis scale by a divisor read from
    its own values, so a uint8 frame and the same frame widened into uint16 land
    in the same place. A set that disagrees with itself - genuine 8-bit pictures
    beside genuine 16-bit ones - has no single answer, and quietly averaging two
    pictures of "normal" that differ 257-fold in brightness produces a profile
    so wide that nothing is ever anomalous again. That is the degenerate result
    the caller has to hear about, so it comes back as a warning and the whole
    set is re-read on the scale most of it already used.

    Returns:
        ``(scale, warning)``. ``scale`` is ``None`` when nothing integer was
        fitted, and ``warning`` is ``None`` when the set agreed.
    """
    seen: Dict[float, int] = {}
    for image in loaded:
        if image.value_scale is not None:
            seen[float(image.value_scale)] = seen.get(float(image.value_scale), 0) + 1
    if not seen:
        return None, None
    # Most images win; a tie goes to the widest scale, which dims an image
    # rather than saturating it, and dimming is the recoverable mistake.
    chosen = max(seen, key=lambda value: (seen[value], value))
    if len(seen) == 1:
        return chosen, None
    shown = ", ".join(
        "0-{0:g} ({1} image{2})".format(value, seen[value], "" if seen[value] == 1 else "s")
        for value in sorted(seen, reverse=True)
    )
    return chosen, (
        "the known-good images do not agree on how deep their pixels are: "
        "{0}. Mixing pixel ranges in one normal set makes the spread of every "
        "feature meaningless, so all of them were re-read on the 0-{1:g} scale. "
        "Convert the set to one bit depth and fit again".format(shown, chosen)
    )


def _distinct_sizes(loaded: Sequence[LoadedImage]) -> List[Tuple[int, int]]:
    """Every source size seen, largest area first."""
    seen: Dict[Tuple[int, int], None] = {}
    for image in loaded:
        seen[image.size] = None
    return sorted(seen, key=lambda size: size[0] * size[1], reverse=True)


def _fit_notes(
    loaded: Sequence[LoadedImage],
    sizes: Sequence[Tuple[int, int]],
    analysis_size: int,
) -> List[str]:
    """What the user should know about how the known-good set was read."""
    notes: List[str] = []
    if any(image.resized for image in loaded):
        notes.append(
            "images are resampled onto a {0}x{0} square before measuring".format(
                analysis_size
            )
        )
    if len(sizes) > 1:
        shown = ", ".join("{0}x{1}".format(w, h) for w, h in sizes[:4])
        more = "" if len(sizes) <= 4 else " and {0} more".format(len(sizes) - 4)
        notes.append(
            "the known-good images came in {0} different sizes ({1}{2}); on the "
            "square analysis grid they are compared as proportions, not "
            "pixels".format(len(sizes), shown, more)
        )
    channels = sorted({image.channels for image in loaded})
    if len(channels) > 1:
        notes.append(
            "the known-good images mix {0}; greyscale is carried in all three "
            "channels and alpha is composited over white, so they are "
            "comparable".format(
                " and ".join(
                    {1: "greyscale", 3: "colour", 4: "colour with alpha"}.get(
                        count, "{0}-channel".format(count)
                    )
                    for count in channels
                )
            )
        )
    for image in loaded:
        for note in image.notes:
            text = "{0}: {1}".format(image.source, note)
            if text not in notes:
                notes.append(text)
    return notes


def _predict_notes(
    loaded: LoadedImage, profile: Profile, analysis_size: int
) -> List[str]:
    """What the user should know about how this one image was read.

    The mixed-size note has to land here, not only on the profile: a caller
    reading ``predict_batch`` results or the CLI's JSON never looks at
    ``detector.summary()``, and "everything was resampled onto one square grid"
    is exactly the caveat that explains a score they are about to act on.
    """
    notes: List[str] = list(loaded.notes)
    if profile.sizes and loaded.size not in profile.sizes:
        shown = ", ".join("{0}x{1}".format(w, h) for w, h in profile.sizes[:3])
        notes.append(
            "this image is {0}x{1}, and the known-good set was {2}; both were "
            "resampled onto the {3}x{3} analysis grid, so they are compared as "
            "proportions, not pixels".format(
                loaded.width, loaded.height, shown, analysis_size
            )
        )
    elif len(profile.sizes) > 1:
        shown = ", ".join("{0}x{1}".format(w, h) for w, h in profile.sizes[:3])
        more = "" if len(profile.sizes) <= 3 else " and {0} more".format(
            len(profile.sizes) - 3
        )
        notes.append(
            "the known-good set came in {0} different sizes ({1}{2}); everything, "
            "this image included, is resampled onto the {3}x{3} analysis grid and "
            "compared as proportions, not pixels".format(
                len(profile.sizes), shown, more, analysis_size
            )
        )
    return notes
