"""Fixtures that build a fake machine: packages, model files, binaries, audio.

Everything here is generated on the spot. No test imports a real engine,
downloads a model or opens a network connection, and every test runs against
an isolated home folder, so the models and engines on the machine running the
suite never leak into the results.
"""

from __future__ import annotations

import io
import math
import os
import pickle
import stat
import struct
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pytest

from offline_stt_router import Machine, Router

ENV_TO_CLEAR = (
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME",
    "VOSK_MODEL_PATH", "WHISPER_CPP_BIN", "WHISPER_CPP_MODELS", "OFFLINE_STT_MODELS",
)

# --------------------------------------------------------------------------- env


class FakeWorld:
    """A home folder, a site-packages folder and a bin folder, all empty."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / "home"
        self.site = root / "site"
        self.bin = root / "bin"
        for folder in (self.home, self.site, self.bin):
            folder.mkdir(parents=True, exist_ok=True)

    # ---- packages ---------------------------------------------------------

    def package(
        self,
        module: str,
        dist: Optional[str] = None,
        version: str = "1.0.0",
        files: Optional[Dict[str, str]] = None,
        init: str = "raise RuntimeError('this package must never be imported by a probe')\n",
    ) -> Path:
        """A package folder plus, if ``dist`` is given, its dist-info metadata."""
        folder = self.site / module
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text(init, encoding="utf-8")
        for name, body in (files or {}).items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if body == "<binary>":
                path.write_bytes(b"\x7fELF fake native library")
            else:
                path.write_text(body, encoding="utf-8")
        if dist:
            self.metadata(dist, version)
        return folder

    def metadata(self, dist: str, version: str) -> Path:
        info = self.site / "{0}-{1}.dist-info".format(dist.replace("-", "_"), version)
        info.mkdir(parents=True, exist_ok=True)
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: {0}\nVersion: {1}\n".format(dist, version), encoding="utf-8"
        )
        return info

    def modules(self, *names: str) -> None:
        """Plain dependency modules that only need to be findable."""
        for name in names:
            (self.site / (name + ".py")).write_text("# dependency stand-in\n", encoding="utf-8")

    def faster_whisper(self, deps: Sequence[str] = ("ctranslate2", "av", "tokenizers", "huggingface_hub", "numpy"),
                       init: Optional[str] = None) -> Path:
        self.modules(*deps)
        if init is None:
            return self.package("faster_whisper", "faster-whisper", "1.1.0")
        return self.package("faster_whisper", "faster-whisper", "1.1.0", init=init)

    def openai_whisper(self, deps: Sequence[str] = ("torch", "numpy", "tiktoken", "numba"),
                       init: Optional[str] = None) -> Path:
        self.modules(*deps)
        kwargs = {} if init is None else {"init": init}
        return self.package("whisper", "openai-whisper", "20240930",
                            files={"transcribe.py": "", "audio.py": ""}, **kwargs)

    def vosk(self, with_library: bool = True, init: Optional[str] = None) -> Path:
        from offline_stt_router._engines import vosk_library_name

        self.modules("_cffi_backend", "requests", "srt", "tqdm")
        files = {vosk_library_name(): "<binary>"} if with_library else {}
        kwargs = {} if init is None else {"init": init}
        return self.package("vosk", "vosk", "0.3.45", files=files, **kwargs)

    # ---- model files ------------------------------------------------------

    @property
    def hub(self) -> Path:
        return self.home / ".cache" / "huggingface" / "hub"

    def ct2_model(self, repo: str, complete: bool = True, languages: int = 99, revision: str = "abc123") -> Path:
        org, name = repo.split("/", 1)
        repo_dir = self.hub / "models--{0}--{1}".format(org, name)
        snapshot = repo_dir / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
        config = {"alignment_heads": [[1, 0]], "lang_ids": list(range(50259, 50259 + languages)),
                  "suppress_ids": [1], "suppress_ids_begin": [220]}
        import json

        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        if complete:
            (snapshot / "model.bin").write_bytes(b"\0" * 4096)
        return snapshot

    def transformers_model(self, repo: str) -> Path:
        org, name = repo.split("/", 1)
        snapshot = self.hub / "models--{0}--{1}".format(org, name) / "snapshots" / "r1"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"\0" * 64)
        return snapshot

    def pt_model(self, folder: Path, name: str, dims: Dict[str, int], truncate: bool = False) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (name + ".pt")
        write_pt(path, dims, truncate=truncate)
        return path

    def ggml_model(self, folder: Path, name: str, **header: int) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "ggml-{0}.bin".format(name)
        write_ggml(path, **header)
        return path

    def vosk_model(self, folder: Path, name: str, graph: bool = True) -> Path:
        model = folder / name
        (model / "am").mkdir(parents=True, exist_ok=True)
        (model / "conf").mkdir(parents=True, exist_ok=True)
        (model / "am" / "final.mdl").write_bytes(b"\0" * 2048)
        (model / "conf" / "mfcc.conf").write_text("--sample-frequency=16000\n", encoding="utf-8")
        if graph:
            (model / "graph").mkdir(exist_ok=True)
            (model / "graph" / "HCLr.fst").write_bytes(b"\0" * 1024)
        return model

    # ---- binaries ---------------------------------------------------------

    def script(self, name: str, python_code: str) -> Path:
        """An executable called ``name`` on the fake PATH that runs ``python_code``."""
        body = self.bin / (name + "_impl.py")
        body.write_text(textwrap.dedent(python_code), encoding="utf-8")
        if os.name == "nt":
            launcher = self.bin / (name + ".cmd")
            launcher.write_text('@"{0}" "{1}" %*\r\n'.format(sys.executable, body), encoding="utf-8")
        else:
            launcher = self.bin / name
            launcher.write_text('#!/bin/sh\nexec "{0}" "{1}" "$@"\n'.format(sys.executable, body), encoding="utf-8")
            launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return launcher

    # ---- routers ----------------------------------------------------------

    def router(self, machine: Optional[Machine] = None, **kw) -> Router:
        return Router(machine=machine or roomy_machine(), search_paths=[str(self.site)], **kw)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    """An isolated machine with nothing installed and no models anywhere."""
    fake = FakeWorld(tmp_path)
    for name in ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(fake.home))
    monkeypatch.setenv("USERPROFILE", str(fake.home))
    monkeypatch.setenv("PATH", str(fake.bin))
    monkeypatch.setenv("OFFLINE_STT_ROUTER_NO_GPU", "1")
    return fake


@pytest.fixture
def importable(world: FakeWorld, monkeypatch: pytest.MonkeyPatch):
    """Put the fake site folder on sys.path, and forget fake modules afterwards."""
    monkeypatch.syspath_prepend(str(world.site))
    before = set(sys.modules)
    yield world
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", "") or ""
        if origin.startswith(str(world.site)):
            del sys.modules[name]


def roomy_machine(**kw) -> Machine:
    """A 16 GB, 8-core machine with no GPU (the common laptop case)."""
    values = dict(os="Linux", arch="x86_64", cpu_logical=8, cpu_physical=8,
                  ram_total_gb=16.0, ram_available_gb=12.0)
    values.update(kw)
    return Machine(**values)


# --------------------------------------------------------------------------- model bytes

TINY_DIMS = dict(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6,
                 n_audio_layer=4, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4)


def write_ggml(path: Path, n_vocab: int = 51865, n_audio_state: int = 16, n_audio_layer: int = 2,
               n_text_state: int = 16, n_text_layer: int = 2, n_mels: int = 80, ftype: int = 1,
               pad_to: Optional[int] = None) -> Path:
    """A whisper.cpp header with small dimensions, followed by enough bytes to be whole."""
    header = struct.pack("<I", 0x67676D6C) + struct.pack(
        "<11i", n_vocab, 1500, n_audio_state, 4, n_audio_layer, 448, n_text_state, 4,
        n_text_layer, n_mels, ftype,
    )
    from offline_stt_router._models import _GGML_FTYPES, _weight_count

    dims = dict(n_vocab=n_vocab, n_audio_state=n_audio_state, n_audio_layer=n_audio_layer,
                n_text_state=n_text_state, n_text_layer=n_text_layer, n_mels=n_mels)
    expected = int(_weight_count(dims) * _GGML_FTYPES[ftype % 1000][1])
    size = expected + 1024 if pad_to is None else pad_to
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(b"\0" * max(0, size - len(header)))
    return path


def write_pt(path: Path, dims: Dict[str, int], truncate: bool = False) -> Path:
    """A zip checkpoint laid out like torch.save, holding a plain pickled dict."""
    payload = pickle.dumps({"dims": dict(dims), "model_state_dict": {}}, protocol=2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("checkpoint/data.pkl", payload)
        archive.writestr("checkpoint/version", "3\n")
        archive.writestr("checkpoint/data/0", b"\0" * 8192)
    data = buffer.getvalue()
    if truncate:
        data = data[: len(data) // 2]
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- audio


def tone(freq: float = 440.0, seconds: float = 1.0, rate: int = 16000, level: float = 0.5) -> List[float]:
    """A sine wave: the stand-in for speech wherever only the plumbing is tested."""
    count = int(seconds * rate)
    return [level * math.sin(2 * math.pi * freq * i / rate) for i in range(count)]


def write_wav_bytes(samples: Iterable[float], rate: int = 16000, bits: int = 16, channels: int = 1,
                    float_format: bool = False, extensible: bool = False, extra_chunk: bool = False) -> bytes:
    """Encode ``samples`` (mono) as a WAV of the given layout, every channel the same."""
    values = list(samples)
    frames = bytearray()
    for value in values:
        for _ in range(channels):
            if float_format:
                frames += struct.pack("<f" if bits == 32 else "<d", value)
            elif bits == 8:
                frames += struct.pack("<B", max(0, min(255, int(round(value * 128)) + 128)))
            elif bits == 16:
                frames += struct.pack("<h", max(-32768, min(32767, int(round(value * 32768)))))
            elif bits == 24:
                number = max(-8388608, min(8388607, int(round(value * 8388608))))
                frames += struct.pack("<i", number)[:3]
            else:
                frames += struct.pack("<i", max(-2147483648, min(2147483647, int(round(value * 2147483648)))))
    tag = 3 if float_format else 1
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * block, block, bits)
        guid_tail = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", tag) + guid_tail
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if extra_chunk:
        note = b"odd"  # odd length: must be padded to an even boundary
        body += b"LIST" + struct.pack("<I", len(note)) + note + b"\0"
    body += b"data" + struct.pack("<I", len(frames)) + bytes(frames)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_wav(path: Path, samples: Iterable[float], **kw) -> Path:
    path.write_bytes(write_wav_bytes(samples, **kw))
    return path


def zero_crossing_frequency(samples: Sequence[float], rate: int) -> float:
    """The frequency of a pure tone, from how often it crosses zero."""
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0 <= b) or (a >= 0 > b))
    return crossings / 2.0 / (len(samples) / float(rate))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch):
    """Make every way of opening a connection raise, and count the attempts."""
    import socket
    import urllib.request

    attempts: List[str] = []

    def blocked(*args, **kwargs):
        attempts.append(repr(args[:2]))
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    return attempts
