

def _scene():
    import numpy as np

    height, width = 400, 600
    scene = np.zeros((height, width, 3), np.uint8)
    for x in range(0, width, 40):
        scene[:, x:x + 20] = 220
    for y in range(0, height, 40):
        scene[y:y + 20, :] = np.clip(scene[y:y + 20, :].astype(int) - 120, 0, 255)
    return scene


def _smear(array, width, axis):
    import numpy as np

    stack = array.astype(float)
    out = np.zeros_like(stack)
    for step in range(width):
        out += np.roll(stack, step - width // 2, axis=axis)
    return (out / width).astype(np.uint8)


def test_motion_blur_is_noticed_at_all():
    """Regression: an isotropic Laplacian cannot see a sideways smear, because every
    vertical edge survives it. A heavily motion-blurred frame scored 94 and graded A -
    HIGHER than the unblurred original - and passed as usable."""
    from PIL import Image

    import image_quality_ai as iq

    scene = _scene()
    sharp = iq.assess(Image.fromarray(scene))
    smeared = iq.assess(Image.fromarray(_smear(scene, 31, 1)))

    assert smeared.score < sharp.score, "a smeared frame must not outscore the original"
    assert "one way" in smeared.metrics["sharpness"].message, smeared.metrics["sharpness"].message
    assert smeared.metrics["sharpness"].details["directional_blur"] is True
    assert sharp.metrics["sharpness"].details["directional_blur"] is False


def test_a_subject_that_varies_in_one_direction_is_not_condemned():
    """Crisp vertical stripes leave one axis featureless while being perfectly sharp, and
    a still frame cannot tell that apart from a smear. It is flagged, not failed."""
    import numpy as np
    from PIL import Image

    import image_quality_ai as iq

    stripes = np.zeros((400, 600, 3), np.uint8)
    for x in range(0, 600, 24):
        stripes[:, x:x + 12] = 240

    report = iq.assess(Image.fromarray(stripes))
    assert report.usable is True, "a sharp striped subject must not be called unusable"
    assert "one way" in report.metrics["sharpness"].message


def test_an_out_of_focus_frame_is_still_failed():
    from PIL import Image

    import image_quality_ai as iq

    sharp = Image.fromarray(_scene())
    soft = sharp.resize((50, 33)).resize((600, 400))
    report = iq.assess(soft)
    assert report.usable is False
    assert iq.is_blurry(soft) is True
