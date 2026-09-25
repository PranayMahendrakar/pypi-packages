"""Model files: identified from their own bytes, partial downloads caught."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from offline_stt_router._models import (
    LocalModel,
    family_from_name,
    inspect_ct2,
    inspect_ggml,
    inspect_path,
    inspect_pt,
    inspect_vosk,
    read_pt_dims,
    scan_models,
)

from conftest import TINY_DIMS, write_ggml, write_pt


@pytest.mark.parametrize(
    "name, family",
    [
        ("faster-whisper-large-v3", "large-v3"),
        ("faster-whisper-large-v3-turbo", "large-v3-turbo"),
        ("large-v2", "large-v2"),
        ("ggml-base.en", "base"),
        ("small.en", "small"),
        ("tiny", "tiny"),
        ("faster-distil-whisper-large-v3", "distil-large-v3"),
        ("distil-large-v3.5-ct2", "distil-large-v3.5"),
        ("distil-medium.en", "distil-medium"),
        ("whisper-large", "large-v3"),
        ("database-model", None),
        ("", None),
    ],
)
def test_family_from_name(name, family):
    assert family_from_name(name) == family


# --------------------------------------------------------------------------- ggml


def test_ggml_complete_file_is_read_from_its_header(tmp_path):
    path = write_ggml(tmp_path / "ggml-base.en.bin", n_vocab=51864, ftype=1)
    model, note = inspect_ggml(str(path))
    assert note is None
    assert model.name == "base.en" and model.family == "base"
    assert model.languages == ["en"]  # English-only read from the vocabulary size
    assert model.quantization == "f16"


def test_ggml_header_beats_a_misleading_name(tmp_path):
    # real "tiny" dimensions, filed under a name that says large
    path = tmp_path / "ggml-large-v3.bin"
    write_ggml(path, n_audio_state=384, n_audio_layer=4, n_text_state=384, n_text_layer=4, ftype=2)
    model, note = inspect_ggml(str(path))
    assert note is None
    assert model.family == "tiny"
    assert model.quantization == "q4_0"
    assert len(model.languages) == 99


def test_ggml_truncated_file_is_an_interrupted_download(tmp_path):
    whole = write_ggml(tmp_path / "ggml-small.bin")
    size = whole.stat().st_size
    cut = tmp_path / "ggml-cut.bin"
    cut.write_bytes(whole.read_bytes()[: size // 3])
    model, note = inspect_ggml(str(cut))
    assert model is None
    assert "cut short" in note and "interrupted download" in note
    # the whole file is still fine: no false alarm
    model, note = inspect_ggml(str(whole))
    assert model is not None and note is None


def test_ggml_bad_header_empty_file_and_gguf(tmp_path):
    junk = tmp_path / "ggml-junk.bin"
    junk.write_bytes(b"not a model at all" * 10)
    empty = tmp_path / "ggml-empty.bin"
    empty.write_bytes(b"")
    gguf = tmp_path / "ggml-new.bin"
    gguf.write_bytes(b"GGUF" + b"\0" * 100)
    assert "model header" in inspect_ggml(str(junk))[1]
    assert "empty" in inspect_ggml(str(empty))[1]
    assert "GGUF" in inspect_ggml(str(gguf))[1]


def test_ggml_v3_turbo_from_dimensions(tmp_path):
    path = tmp_path / "ggml-mystery.bin"
    write_ggml(path, n_vocab=51866, n_audio_state=1280, n_audio_layer=32, n_text_state=1280,
               n_text_layer=4, n_mels=128, ftype=7, pad_to=900 * 1024 * 1024 // 1000)
    model, note = inspect_ggml(str(path))
    # size says cut short (1 MB for a turbo), so it must be refused rather than trusted
    assert model is None and "cut short" in note


# --------------------------------------------------------------------------- .pt


def test_pt_dims_are_read_without_unpickling(tmp_path):
    path = write_pt(tmp_path / "anything.pt", TINY_DIMS)
    dims, status = read_pt_dims(str(path))
    assert status == "ok"
    assert dims["n_audio_state"] == 384 and dims["n_vocab"] == 51865
    model, note = inspect_pt(str(path))
    assert note is None and model.family == "tiny"


def test_pt_english_only_from_vocab(tmp_path):
    dims = dict(TINY_DIMS, n_vocab=51864, n_audio_state=512, n_text_state=512, n_audio_layer=6, n_text_layer=6)
    model, _ = inspect_pt(str(write_pt(tmp_path / "base.en.pt", dims)))
    assert model.family == "base" and model.languages == ["en"]


def test_pt_truncated_download_is_caught(tmp_path):
    path = write_pt(tmp_path / "small.pt", TINY_DIMS, truncate=True)
    model, note = inspect_pt(str(path))
    assert model is None
    assert "interrupted download" in note


def test_pt_zip_that_is_not_whisper(tmp_path):
    import zipfile

    path = tmp_path / "other.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x/data.pkl", b"\x80\x02}q\x00.")
    model, note = inspect_pt(str(path))
    assert model is None and "not a Whisper checkpoint" in note


# --------------------------------------------------------------------------- CTranslate2


def test_ct2_complete_incomplete_and_transformers(world):
    good = world.ct2_model("Systran/faster-whisper-small")
    unfinished = world.ct2_model("Systran/faster-whisper-large-v2", complete=False)
    hf = world.transformers_model("openai/whisper-base")
    model, note = inspect_ct2(str(good), label="small")
    assert note is None and model.family == "small" and len(model.languages) == 99
    model, note = inspect_ct2(str(unfinished), label="large-v2")
    assert model is None and "not finished" in note
    model, note = inspect_ct2(str(hf), label="openai/whisper-base")
    assert model is None and "Transformers" in note


def test_ct2_v3_has_100_languages_and_distil_is_english(world):
    v3 = world.ct2_model("Systran/faster-whisper-large-v3", languages=100)
    distil = world.ct2_model("Systran/faster-distil-whisper-large-v3", languages=100)
    assert "yue" in inspect_ct2(str(v3), label="large-v3")[0].languages
    assert inspect_ct2(str(distil), label="distil-large-v3")[0].languages == ["en"]


def test_ct2_broken_symlink(world, tmp_path):
    snapshot = world.ct2_model("Systran/faster-whisper-tiny", complete=False)
    try:
        os.symlink(str(tmp_path / "missing-blob"), str(snapshot / "model.bin"))
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow symlinks")
    model, note = inspect_ct2(str(snapshot), label="tiny")
    assert model is None and "missing file" in note


def test_hf_scan_names_and_notes(world):
    world.ct2_model("Systran/faster-whisper-small")
    world.ct2_model("mobiuslabsgmbh/faster-whisper-large-v3-turbo", languages=100)
    world.ct2_model("someone/whisper-hindi-small-ct2")
    world.ct2_model("Systran/faster-whisper-medium", complete=False)
    world.transformers_model("openai/whisper-base")
    world.ct2_model("unrelated/bert-base")  # not a whisper repo: ignored entirely
    scan = scan_models()
    names = sorted(m.name for m in scan.for_engine("faster-whisper"))
    assert names == ["large-v3-turbo", "small", "someone/whisper-hindi-small-ct2"]
    notes = " | ".join(scan.notes_for("faster-whisper"))
    assert "medium: download not finished" in notes
    assert "openai/whisper-base" in notes
    assert "bert" not in notes


# --------------------------------------------------------------------------- Vosk


def test_vosk_model_language_and_family(world):
    root = world.home / ".cache" / "vosk"
    small = world.vosk_model(root, "vosk-model-small-en-us-0.15")
    big = world.vosk_model(root, "vosk-model-cn-0.22")
    unknown = world.vosk_model(root, "my-model")
    assert inspect_vosk(str(small))[0].languages == ["en"]
    model = inspect_vosk(str(big))[0]
    assert model.languages == ["zh"] and model.family == "vosk-big"
    model = inspect_vosk(str(unknown))[0]
    assert model.languages == [] and "does not say which language" in model.notes[0]


def test_vosk_incomplete_unzip_and_zip_left_zipped(world):
    root = world.home / ".cache" / "vosk"
    world.vosk_model(root, "vosk-model-small-hi-0.22", graph=False)
    (root / "vosk-model-small-fr-0.22.zip").write_bytes(b"PK\x03\x04")
    world.vosk_model(root, "vosk-model-small-de-0.15")
    scan = scan_models()
    assert [m.name for m in scan.for_engine("vosk")] == ["vosk-model-small-de-0.15"]
    notes = " | ".join(scan.notes_for("vosk"))
    assert "unzip looks incomplete" in notes
    assert "still zipped" in notes


def test_offline_stt_models_folder_holds_every_kind(world, monkeypatch):
    mixed = world.root / "my models"
    write_ggml(mixed / "ggml-base.bin" if mixed.mkdir() is None else None)
    write_pt(mixed / "tiny.pt", TINY_DIMS)
    world.vosk_model(mixed, "vosk-model-small-ru-0.22")
    ct2 = mixed / "my-small-ct2"
    ct2.mkdir()
    (ct2 / "model.bin").write_bytes(b"\0" * 100)
    (ct2 / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OFFLINE_STT_MODELS", str(mixed) + os.pathsep + str(world.root / "missing"))
    scan = scan_models()
    assert [m.name for m in scan.for_engine("whisper.cpp")] == ["base"]
    assert [m.name for m in scan.for_engine("openai-whisper")] == ["tiny"]
    assert [m.name for m in scan.for_engine("vosk")] == ["vosk-model-small-ru-0.22"]
    assert [m.name for m in scan.for_engine("faster-whisper")] == ["my-small-ct2"]
    assert any("does not exist" in n for n in scan.notes_for("any"))


def test_same_file_found_twice_is_listed_once(world, monkeypatch):
    folder = world.home / ".cache" / "whisper.cpp"
    world.ggml_model(folder, "tiny")
    monkeypatch.setenv("WHISPER_CPP_MODELS", str(folder))
    assert len(scan_models().for_engine("whisper.cpp")) == 1


def test_inspect_path_dispatch(world, tmp_path):
    ggml = write_ggml(tmp_path / "ggml-tiny.bin")
    pt = write_pt(tmp_path / "tiny.pt", TINY_DIMS)
    vosk = world.vosk_model(tmp_path, "vosk-model-small-en-us-0.15")
    ct2 = world.ct2_model("Systran/faster-whisper-small")
    assert inspect_path(str(ggml))[0].engine == "whisper.cpp"
    assert inspect_path(str(pt))[0].engine == "openai-whisper"
    assert inspect_path(str(vosk))[0].engine == "vosk"
    assert inspect_path(str(ct2))[0].engine == "faster-whisper"
    assert "does not exist" in inspect_path(str(tmp_path / "nope"))[1]


def test_local_model_helpers():
    model = LocalModel(name="x", engine="faster-whisper", family="small", languages=["en", "hi"])
    assert model.multilingual and model.supports("hi") and not model.supports("fr")
    assert model.supports(None)
    assert model.describe_languages() == "2 languages"
    assert str(model) == "x"
    assert LocalModel(name="v", engine="vosk", family="vosk-small", languages=["en"]).describe_languages() == "English only"
    assert not LocalModel(name="v", engine="vosk", family="vosk-small", languages=["en", "hi"]).supports(None)
    assert model.to_dict()["languages"] == ["en", "hi"]
