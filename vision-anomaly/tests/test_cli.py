"""The command line: exit codes, formats, and surviving a hostile console."""
from __future__ import annotations

import json

import pytest

from conftest import (
    different_scene,
    held_out_scene,
    missing_part,
    normal_scene,
    normal_set,
    save_png,
)
from vision_anomaly.cli import BAD_USAGE, FOUND, OK, build_parser, main


@pytest.fixture(scope="session")
def good_dir(tmp_path_factory):
    """Eight known-good frames on disk. Session scoped: nothing writes to it."""
    folder = tmp_path_factory.mktemp("good")
    for index, image in enumerate(normal_set(8)):
        save_png(image, folder / "good_{0:02d}.png".format(index))
    return str(folder)


@pytest.fixture(scope="session")
def mixed_dir(tmp_path_factory):
    """One clean frame and one obvious stranger, both with non-ASCII names."""
    folder = tmp_path_factory.mktemp("today")
    save_png(held_out_scene(0), folder / "pièce-côté-gauche.png")
    save_png(different_scene(0), folder / "部品-異常.png")
    return str(folder)


def test_help_works_and_says_what_the_tool_is_for(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0

    text = capsys.readouterr().out
    assert "--good" in text
    assert "--profile" in text
    assert "exit codes" in text
    assert text.isascii()


def test_version_prints_the_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "vision-anomaly 0.1.0" in capsys.readouterr().out


def test_a_clean_folder_exits_zero(good_dir, tmp_path, capsys):
    folder = tmp_path / "clean"
    folder.mkdir()
    for index in range(3):
        save_png(held_out_scene(index), folder / "f{0}.png".format(index))

    assert main(["--good", good_dir, str(folder)]) == OK
    assert "0 anomalous" in capsys.readouterr().out


def test_a_stranger_in_the_folder_exits_one(good_dir, mixed_dir, capsys):
    code = main(["--good", good_dir, mixed_dir])

    out = capsys.readouterr().out
    assert code == FOUND
    assert "1 anomalous" in out
    assert "ANOMALOUS" in out


@pytest.mark.parametrize("extra", [[], ["--quiet"], ["--json"]])
def test_non_ascii_names_survive_being_printed(good_dir, mixed_dir, capsys, extra):
    """The failure this guards against lands at print time, after all the work.

    Checked in all three output formats, because each one builds its text a
    different way and only one of them has to forget.
    """
    main(["--good", good_dir, mixed_dir] + extra)

    out = capsys.readouterr().out
    assert "部品-異常" in out
    if extra:
        # The default summary describes only the flagged images in full; the
        # listing formats name every one of them.
        assert "pièce-côté-gauche" in out


def test_json_output_is_parseable_and_keeps_non_ascii(good_dir, mixed_dir, capsys):
    code = main(["--good", good_dir, mixed_dir, "--json"])

    data = json.loads(capsys.readouterr().out)
    assert code == FOUND
    assert data["n_images"] == 2
    assert data["n_anomalous"] == 1
    sources = " ".join(result["source"] for result in data["results"])
    assert "異常" in sources


def test_quiet_gives_one_line_per_image(good_dir, mixed_dir, capsys):
    main(["--good", good_dir, mixed_dir, "--quiet"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert any("anomalous" in line for line in lines)
    assert any("normal" in line for line in lines)


def test_sensitivity_changes_the_verdict(good_dir, tmp_path, capsys):
    folder = tmp_path / "borderline"
    folder.mkdir()
    save_png(missing_part(0), folder / "gone.png")

    assert main(["--good", good_dir, str(folder)]) == FOUND
    capsys.readouterr()
    assert main(["--good", good_dir, str(folder), "--sensitivity", "500"]) == OK


def test_a_profile_can_be_saved_and_reused(good_dir, mixed_dir, tmp_path, capsys):
    profile = str(tmp_path / "belt.json")

    assert main(["--good", good_dir, "--save-profile", profile]) == OK
    assert "profile written" in capsys.readouterr().err
    assert json.loads(open(profile, encoding="utf-8").read())["n_images"] == 8

    assert main(["--profile", profile, mixed_dir]) == FOUND
    assert "1 anomalous" in capsys.readouterr().out


def test_output_is_written_as_utf8(good_dir, mixed_dir, tmp_path, capsys):
    target = tmp_path / "reports" / "today.json"

    main(["--good", good_dir, mixed_dir, "--json", "--output", str(target)])
    capsys.readouterr()

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["n_anomalous"] == 1


def test_recursive_walks_subfolders(good_dir, tmp_path, capsys):
    folder = tmp_path / "tree"
    (folder / "monday").mkdir(parents=True)
    save_png(held_out_scene(0), folder / "top.png")
    save_png(different_scene(0), folder / "monday" / "deep.png")

    assert main(["--good", good_dir, str(folder)]) == OK
    capsys.readouterr()
    assert main(["--good", good_dir, str(folder), "--recursive"]) == FOUND


def test_a_weak_profile_warns_on_stderr(tmp_path, capsys):
    folder = tmp_path / "few"
    folder.mkdir()
    for index, image in enumerate(normal_set(3)):
        save_png(image, folder / "g{0}.png".format(index))
    check = tmp_path / "check"
    check.mkdir()
    save_png(held_out_scene(0), check / "one.png")

    main(["--good", str(folder), str(check)])

    assert "only 3 images" in capsys.readouterr().err


def test_describe_features_prints_the_table(capsys):
    assert main(["--describe-features"]) == OK

    text = capsys.readouterr().out
    assert "216 numbers per image" in text
    assert "orientation" in text


@pytest.mark.parametrize(
    "argv",
    [
        [],                                            # nothing at all
        ["images/"],                                   # no normal set
        ["--good", "a", "--profile", "b", "x"],        # both sources
        ["--good", "a"],                               # nothing to check, nothing to save
    ],
)
def test_unusable_arguments_exit_two(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == BAD_USAGE


def test_a_missing_folder_exits_two_with_a_clear_message(good_dir, tmp_path, capsys):
    code = main(["--good", good_dir, str(tmp_path / "nowhere")])

    assert code == BAD_USAGE
    assert "no such image or directory" in capsys.readouterr().err


def test_a_corrupt_image_exits_two_naming_the_file(good_dir, tmp_path, capsys):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "broken.png").write_bytes(b"not an image")

    code = main(["--good", good_dir, str(folder)])

    assert code == BAD_USAGE
    assert "broken.png" in capsys.readouterr().err


def test_a_missing_profile_exits_two(tmp_path, capsys):
    code = main(["--profile", str(tmp_path / "gone.json"), str(tmp_path)])

    assert code == BAD_USAGE
    assert "no such profile file" in capsys.readouterr().err


def test_the_parser_renders_without_a_terminal():
    """argparse formatting is a real source of crashes on odd consoles."""
    text = build_parser().format_help()

    assert text.isascii()
    assert "--describe-features" in text


@pytest.fixture(scope="session")
def mixed_size_dir(tmp_path_factory):
    """Six known-good frames, every one a different shape."""
    folder = tmp_path_factory.mktemp("mixedsize")
    shapes = [(320, 240), (256, 256), (400, 300), (200, 260), (512, 384), (300, 300)]
    for index, size in enumerate(shapes):
        save_png(normal_scene(index, size), folder / "m{0}.png".format(index))
    return str(folder)


def test_the_resize_caveat_is_printed_even_when_nothing_is_flagged(
    mixed_size_dir, capsys
):
    """It used to reach only detector.summary(), which the CLI never prints."""
    code = main(["--good", mixed_size_dir, mixed_size_dir])

    text = capsys.readouterr().out
    assert code == OK
    assert "note:" in text
    assert "different sizes" in text


def test_the_resize_caveat_reaches_json_consumers(mixed_size_dir, capsys):
    main(["--good", mixed_size_dir, mixed_size_dir, "--json"])

    data = json.loads(capsys.readouterr().out)

    assert data["results"][0]["notes"]
    assert any("different sizes" in note for note in data["results"][0]["notes"])


def test_a_missing_known_good_directory_still_names_itself(good_dir, tmp_path, capsys):
    code = main(["--good", str(tmp_path / "nosuchdir"), good_dir])

    assert code == BAD_USAGE
    assert "no such known-good image or directory" in capsys.readouterr().err
