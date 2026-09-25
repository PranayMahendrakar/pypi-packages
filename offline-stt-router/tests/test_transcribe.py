"""transcribe(): each engine gets a local model and audio it can read.

The engines here are stand-ins written by the tests: they record what they
were given and refuse to "download" (anything that is not a local path makes
them fail loudly), so these tests check the real calling code end to end.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys

import pytest

import offline_stt_router as stt
from offline_stt_router import GPU, TranscriptionFailed
from offline_stt_router._audio import read_wav

from conftest import TINY_DIMS, roomy_machine, tone, write_wav, zero_crossing_frequency

FAKE_FASTER_WHISPER = '''
import os
CALLS = []

class Segment:
    def __init__(self, text):
        self.text = text

class WhisperModel:
    def __init__(self, model_size_or_path, device="auto", compute_type="default", cpu_threads=0, **kw):
        if not os.path.isdir(model_size_or_path):
            raise AssertionError("asked to download " + str(model_size_or_path))
        if device == "cuda" and os.environ.get("FAKE_CUDA_BROKEN"):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        CALLS.append(("load", model_size_or_path, device, compute_type, cpu_threads))

    def transcribe(self, audio, language=None, beam_size=5, **kw):
        CALLS.append(("transcribe", audio if isinstance(audio, str) else len(audio), language, beam_size))
        def segments():
            yield Segment(" hello")
            yield Segment(" world ")
        return segments(), {"language": language}
'''

FAKE_OPENAI_WHISPER = '''
import os
CALLS = []

class _Model:
    def __init__(self, path, device):
        self.path, self.device = path, device

    def transcribe(self, audio, language=None, fp16=True, **kw):
        CALLS.append((type(audio).__name__, len(audio), language, fp16))
        return {"text": " heard %d samples in %s " % (len(audio), language)}

def load_model(name, device=None, download_root=None, in_memory=False):
    if not os.path.isfile(name):
        raise AssertionError("asked to download " + str(name))
    return _Model(name, device)
'''

FAKE_NUMPY = '''
import array
float32 = "float32"

class _Array(list):
    def copy(self):
        return _Array(self)

def frombuffer(buffer, dtype=None):
    values = array.array("f")
    values.frombytes(bytes(buffer))
    return _Array(values)
'''

FAKE_VOSK = '''
import json, os
LEVELS = []

def SetLogLevel(level):
    LEVELS.append(level)

class Model:
    def __init__(self, model_path=None, model_name=None, lang=None):
        if model_path is None or not os.path.isdir(model_path):
            raise AssertionError("asked to download a model")
        self.path = model_path

class KaldiRecognizer:
    def __init__(self, model, rate):
        self.rate, self.bytes, self.chunks = rate, 0, 0

    def AcceptWaveform(self, data):
        self.bytes += len(data)
        self.chunks += 1
        return self.chunks % 3 == 0

    def Result(self):
        return json.dumps({"text": "part"})

    def FinalResult(self):
        return json.dumps({"text": "total %d bytes at %d" % (self.bytes, self.rate)})
'''

FAKE_WHISPER_CLI = '''
import os, struct, sys
args = sys.argv[1:]

def value(flag):
    return args[args.index(flag) + 1]

assert "-otxt" in args and "-nt" in args, args
if os.environ.get("FAKE_WCPP_FAIL"):
    sys.stderr.write("whisper_init: failed to load model\\n")
    sys.exit(3)
data = open(value("-f"), "rb").read()
at = data.index(b"fmt ") + 8
tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[at:at + 16])
with open(value("-of") + ".txt", "w", encoding="utf-8") as handle:
    handle.write("\\u0928\\u092e\\u0938\\u094d\\u0924\\u0947 %d Hz %d ch %d bit %s t%s\\n"
                 % (rate, channels, bits, value("-l"), value("-t")))
'''


def speech_file(tmp_path, name="talk.wav", seconds=1.0, rate=16000, **layout):
    return write_wav(tmp_path / name, tone(440, seconds, rate), rate=rate, **layout)


@pytest.fixture
def fake_fw(importable):
    importable.faster_whisper(init=FAKE_FASTER_WHISPER)
    importable.ct2_model("Systran/faster-whisper-small")
    return importable


def fw_calls():
    return sys.modules["faster_whisper"].CALLS


# --------------------------------------------------------------------------- nothing installed


def test_nothing_installed_raises_one_clear_error(world, tmp_path):
    audio = speech_file(tmp_path)
    with pytest.raises(TranscriptionFailed) as caught:
        world.router().transcribe(str(audio))
    assert isinstance(caught.value, RuntimeError)
    message = str(caught.value)
    assert "cannot transcribe" in message and "pip install faster-whisper" in message
    assert caught.value.choice is not None and not caught.value.choice.ok


def test_bad_audio_is_refused_before_anything_runs(world, tmp_path):
    csv = tmp_path / "u.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like an audio file"):
        world.router().transcribe(str(csv))
    with pytest.raises(FileNotFoundError):
        world.router().transcribe(str(tmp_path / "missing.wav"))


# --------------------------------------------------------------------------- custom engines


def test_custom_engine_receives_the_audio_it_was_given(world, tmp_path):
    router = world.router()
    heard = {}

    def listen(audio, *, sample_rate=None, language=None):
        heard.update(language=language, rate=sample_rate)
        return "tone at {0:.0f} Hz".format(round(zero_crossing_frequency(audio, sample_rate), -1))

    router.register("ears", lambda: {"quality": 0.9}, listen)
    text = router.transcribe(tone(440, 0.5, 8000), sample_rate=8000, language="fr")
    assert text == "tone at 440 Hz"
    assert heard == {"language": "fr", "rate": 8000}


def test_custom_engine_with_a_file_and_a_minimal_signature(world, tmp_path):
    router = world.router()
    router.register("path-reader", lambda: True, lambda audio: "read {0}".format(len(read_wav(audio)[0])))
    assert router.transcribe(str(speech_file(tmp_path))) == "read 16000"


def test_custom_results_are_coerced_or_refused(world, tmp_path):
    router = world.router()
    router.register("dicty", lambda: {"quality": 0.9}, lambda audio: {"text": " from a dict "})
    assert router.transcribe([0.1] * 100) == "from a dict"
    bad = world.router()
    bad.register("numbers", lambda: True, lambda audio: 42)
    with pytest.raises(TranscriptionFailed) as caught:
        bad.transcribe([0.1] * 100)
    assert "expected a string" in caught.value.attempts[0].error


def test_failure_falls_back_to_the_next_option(world, tmp_path):
    router = world.router()

    def explode(audio):
        raise RuntimeError("GPU fell off the bus")

    router.register("first", lambda: {"quality": 0.95}, explode)
    router.register("second", lambda: {"quality": 0.5}, lambda audio: "second one worked")
    result = router.transcribe_detailed([0.2] * 1600)
    assert result.text == "second one worked" and result.engine == "second"
    assert [(a.engine, a.ok) for a in result.attempts] == [("first", False), ("second", True)]
    assert "GPU fell off the bus" in result.attempts[0].error
    assert str(result) == "second one worked"
    json.dumps(result.to_dict())
    with pytest.raises(TranscriptionFailed) as caught:
        router.transcribe([0.2] * 1600, fallback=False)
    assert "GPU fell off the bus" in str(caught.value)


def test_engine_without_a_transcribe_function_is_chosen_but_not_run(world):
    router = world.router()
    router.register("advice-only", lambda: {"quality": 0.9})
    assert router.choose().engine == "advice-only"
    with pytest.raises(TranscriptionFailed, match="no transcribe function"):
        router.transcribe([0.1] * 10)


def test_top_level_transcribe_with_a_registered_engine():
    stt.register("unit-echo", lambda: {"quality": 1.0, "realtime_factor": 0.001, "ram_gb": 0.01},
                 lambda audio, language=None: "echo {0} {1}".format(len(audio), language))
    try:
        assert stt.transcribe([0.0] * 320, engine="unit-echo", language="hi") == "echo 320 hi"
    finally:
        stt.unregister("unit-echo")


# --------------------------------------------------------------------------- faster-whisper


def test_faster_whisper_is_given_a_local_folder_never_a_name(fake_fw, tmp_path, no_network):
    router = fake_fw.router()
    audio = speech_file(tmp_path)
    assert router.transcribe(str(audio), beam_size=2) == "hello world"
    load, call = fw_calls()
    assert load[0] == "load" and load[2:] == ("cpu", "int8", 8)
    assert load[1].endswith("abc123")  # the snapshot folder in the Hugging Face cache
    assert call == ("transcribe", str(audio.resolve()), "en", 2)
    assert no_network == []


def test_the_last_model_stays_loaded_until_unload(fake_fw, tmp_path):
    router = fake_fw.router()
    audio = str(speech_file(tmp_path))
    router.transcribe(audio)
    router.transcribe(audio)
    assert [c[0] for c in fw_calls()].count("load") == 1
    router.unload()
    router.transcribe(audio)
    assert [c[0] for c in fw_calls()].count("load") == 2


def test_cuda_failure_retries_on_the_cpu(fake_fw, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CUDA_BROKEN", "1")
    gpu = GPU(name="Test GPU", vendor="nvidia", backend="cuda", vram_total_gb=8, vram_free_gb=8)
    router = fake_fw.router(machine=roomy_machine(gpus=[gpu]))
    assert router.choose().device == "cuda"
    result = router.transcribe_detailed(str(speech_file(tmp_path)))
    assert result.text == "hello world"
    assert [c[2] for c in fw_calls() if c[0] == "load"] == ["cpu"]


def test_forcing_a_model_that_is_not_on_disk(fake_fw, tmp_path):
    with pytest.raises(TranscriptionFailed) as caught:
        fake_fw.router().transcribe(str(speech_file(tmp_path)), model="large-v3")
    assert "large-v3" in str(caught.value)


# --------------------------------------------------------------------------- openai-whisper


@pytest.fixture
def fake_openai(importable):
    real_numpy = importlib.util.find_spec("numpy") is not None
    if real_numpy:
        import numpy  # noqa: F401  - cached now, so the stand-in below never shadows it
    importable.openai_whisper(init=FAKE_OPENAI_WHISPER)
    if not real_numpy:
        (importable.site / "numpy.py").write_text(FAKE_NUMPY, encoding="utf-8")
    importable.pt_model(importable.home / ".cache" / "whisper", "tiny", TINY_DIMS)
    return importable


def test_openai_whisper_reads_a_wav_without_ffmpeg(fake_openai, tmp_path):
    audio = speech_file(tmp_path, rate=22050, seconds=0.5, bits=24)
    text = fake_openai.router().transcribe(str(audio), language="Hindi")
    assert text == "heard 8000 samples in hi"  # decoded here and resampled to 16 kHz
    [(_kind, count, language, fp16)] = sys.modules["whisper"].CALLS
    assert (count, language, fp16) == (8000, "hi", False)


def test_openai_whisper_needs_ffmpeg_for_mp3(fake_openai, tmp_path):
    mp3 = tmp_path / "talk.mp3"
    mp3.write_bytes(b"ID3\x04\x00" + b"\0" * 64)
    with pytest.raises(TranscriptionFailed) as caught:
        fake_openai.router().transcribe(str(mp3))
    assert "ffmpeg" in caught.value.attempts[0].error


# --------------------------------------------------------------------------- whisper.cpp


@pytest.fixture
def fake_cpp(world):
    world.script("whisper-cli", FAKE_WHISPER_CLI)
    world.ggml_model(world.home / ".cache" / "whisper.cpp", "base")
    return world


def test_whisper_cpp_gets_a_16k_mono_16bit_wav(fake_cpp, tmp_path, no_network):
    stereo_float = speech_file(tmp_path, "stereo.wav", seconds=0.3, rate=44100, bits=32,
                               float_format=True, channels=2)
    text = fake_cpp.router().transcribe(str(stereo_float), language="hi")
    assert text == "नमस्ते 16000 Hz 1 ch 16 bit hi t8"
    assert no_network == []


def test_whisper_cpp_failure_is_reported_then_falls_back(fake_cpp, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WCPP_FAIL", "1")
    router = fake_cpp.router()
    with pytest.raises(TranscriptionFailed) as caught:
        router.transcribe(str(speech_file(tmp_path)))
    assert "exited with code 3" in caught.value.attempts[0].error
    assert "failed to load model" in caught.value.attempts[0].error
    router.register("backup", lambda: {"quality": 0.1, "realtime_factor": 9.0}, lambda audio: "backup")
    result = router.transcribe_detailed(str(speech_file(tmp_path)))
    assert result.engine == "backup" and not result.attempts[0].ok


def test_forcing_a_model_by_path(fake_cpp, tmp_path):
    other = fake_cpp.ggml_model(tmp_path / "elsewhere", "tiny.en", n_vocab=51864)
    router = fake_cpp.router()
    result = router.transcribe_detailed(str(speech_file(tmp_path)), model=str(other))
    assert result.model == "tiny.en" and result.engine == "whisper.cpp"
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"nope" * 20)
    with pytest.raises(ValueError, match="not a model"):
        router.transcribe(str(speech_file(tmp_path)), model=str(junk))


# --------------------------------------------------------------------------- vosk


def test_vosk_gets_16k_16bit_pcm(importable, tmp_path):
    importable.vosk(init=FAKE_VOSK)
    importable.vosk_model(importable.home / ".cache" / "vosk", "vosk-model-small-en-us-0.15")
    audio = speech_file(tmp_path, seconds=1.0, rate=8000, bits=8)
    result = importable.router().transcribe_detailed(str(audio), engine="vosk")
    assert result.text.endswith("total 32000 bytes at 16000")
    assert result.text.startswith("part")
    assert sys.modules["vosk"].LEVELS == [-1]


def test_forcing_an_engine_that_is_not_ready(world, tmp_path):
    with pytest.raises(TranscriptionFailed, match="Vosk is not usable: not installed") as caught:
        world.router().transcribe(str(speech_file(tmp_path)), engine="vosk")
    assert "pip install vosk" in str(caught.value)
    assert "faster-whisper" not in str(caught.value)
    with pytest.raises(TranscriptionFailed, match="No engine called 'nonesuch' is known") as caught:
        world.router().transcribe(str(speech_file(tmp_path)), engine="nonesuch")
    assert "To fix" not in str(caught.value)
