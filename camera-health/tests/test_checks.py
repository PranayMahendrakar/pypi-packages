

def _wall(seed=0):
    """A camera pointed at a plain painted wall: featureless, but nothing is wrong."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.clip(np.full((240, 320, 3), 150.0) + rng.normal(0, 3, (240, 320, 3)), 0, 255).astype("uint8")


def _room(seed=0):
    """A realistic view: structure at several scales, detail in every part of the frame.

    Neither a flat field nor pure noise - both of those are legitimately reported, so
    neither makes a fair "nothing is wrong" fixture.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    height, width = 240, 320
    y, x = np.mgrid[0:height, 0:width]
    scene = 90 + 40 * np.sin(x / 17.0) + 25 * np.cos(y / 11.0)      # large structure
    scene += 18 * np.sin((x + y) / 5.0)                              # fine texture
    scene[70:150, 90:210] += 45                                      # an object in view
    scene = scene[:, :, None].repeat(3, axis=2)
    scene += rng.normal(0, 2.0, scene.shape)                         # sensor noise
    return np.clip(scene, 0, 255).astype("uint8")


def test_a_featureless_view_says_what_it_saw_rather_than_asserting_a_cause():
    """A covered lens and a camera aimed at a blank wall are identical to a single
    frame. The fault is still raised, because a flat view really can be a covered lens,
    but the message must describe the measurement rather than claim to know why."""
    from PIL import Image

    import camera_health as ch

    faults = ch.check(Image.fromarray(_wall())).faults
    obstruction = [f for f in faults if f.kind == "obstruction"]
    assert obstruction, "a completely flat view is worth reporting"
    message = obstruction[0].message.lower()
    assert "featureless" in message or "no detail" in message


def test_a_normal_view_is_not_called_obstructed():
    from PIL import Image

    import camera_health as ch

    faults = [f.kind for f in ch.check(Image.fromarray(_room())).faults]
    assert "obstruction" not in faults, faults


def test_a_partly_covered_lens_is_still_caught():
    from PIL import Image

    import camera_health as ch

    covered = _room().copy()
    covered[:, :150] = 14
    assert [f for f in ch.check(Image.fromarray(covered)).faults if f.kind == "obstruction"]


def test_a_static_scene_is_not_a_frozen_feed():
    """The trap this package exists to get right: a camera watching an empty corridor
    legitimately produces near-identical frames, and calling that frozen would make the
    monitor useless exactly where it is most often deployed."""
    from PIL import Image

    import camera_health as ch

    static = ch.check_stream([Image.fromarray(_room(i)) for i in range(12)])
    assert not [f for f in static.faults if "froz" in str(f).lower()]

    one = Image.fromarray(_room(0))
    frozen = ch.check_stream([one] * 12)
    assert [f for f in frozen.faults if "froz" in str(f).lower()]
