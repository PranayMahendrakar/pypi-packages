"""The small numpy image operations the rules stand on."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

import synth
from video_event_detector._imageops import box_iou, clean, label, phase_correlate, shift_image


def _flood(mask):
    rows, cols = mask.shape
    out = -np.ones(mask.shape, int)
    count = 0
    for y in range(rows):
        for x in range(cols):
            if mask[y, x] and out[y, x] < 0:
                queue = deque([(y, x)])
                out[y, x] = count
                while queue:
                    a, b = queue.popleft()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            c, d = a + da, b + db
                            if 0 <= c < rows and 0 <= d < cols and mask[c, d] and out[c, d] < 0:
                                out[c, d] = count
                                queue.append((c, d))
                count += 1
    return out, count


def test_label_matches_a_flood_fill():
    rng = np.random.default_rng(0)
    for _ in range(60):
        mask = rng.random((int(rng.integers(1, 25)), int(rng.integers(1, 25)))) < rng.random()
        labels, blobs = label(mask)
        reference, count = _flood(mask)
        assert len(blobs) == count
        for blob in blobs:
            ys, xs = np.nonzero(labels == blob.label)
            assert blob.area == ys.size
            assert (blob.top, blob.bottom, blob.left, blob.right) == (ys.min(), ys.max(), xs.min(), xs.max())
            assert len(set(reference[labels == blob.label].tolist())) == 1


def test_label_empty_mask():
    labels, blobs = label(np.zeros((5, 7), bool))
    assert blobs == [] and (labels == -1).all()


@pytest.mark.parametrize("shift", [(3.0, 2.0), (-2.0, 4.0), (1.5, -2.5)])
def test_phase_correlation_recovers_a_global_shift(shift):
    big = synth.texture(seed=11, margin=10)
    moved, _ = shift_image(big.astype(np.float32), *shift)
    reference = big[10:130, 10:170].astype(np.float32)
    image = moved[10:130, 10:170]
    dy, dx, peak = phase_correlate(reference, image)
    assert dy == pytest.approx(shift[0], abs=0.15)
    assert dx == pytest.approx(shift[1], abs=0.15)
    assert peak > 0.1


def test_phase_correlation_of_flat_images_is_no_match():
    flat = np.full((32, 32), 50.0, np.float32)
    assert phase_correlate(flat, flat) == (0.0, 0.0, 0.0)


def test_shift_image_marks_the_uncovered_border():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    moved, valid = shift_image(image, 2.0, 0.0)
    assert np.array_equal(moved[5], image[3])
    assert not valid[:3].any() and valid[4:6, 4:6].all()


def test_clean_drops_speckle_and_keeps_blocks():
    mask = np.zeros((20, 20), bool)
    mask[2, 2] = True
    mask[8:15, 8:15] = True
    cleaned = clean(mask)
    assert not cleaned[2, 2] and cleaned[8:15, 8:15].all()
    assert box_iou((0, 0, 9, 9), (0, 0, 9, 9)) == 1.0
    assert box_iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0
