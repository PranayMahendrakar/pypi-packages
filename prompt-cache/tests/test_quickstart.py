"""The README quickstart, run exactly as a reader would run it."""

from __future__ import annotations

import prompt_cache


def test_readme_quickstart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @prompt_cache.cached(".demo-cache")
    def ask(prompt, model="gpt-4o"):
        return "pretend this call cost you money: " + prompt

    first = ask("Why is the sky blue?")
    second = ask("Why is the sky blue?")
    summary = ask.cache.stats().summary()

    assert first == second == "pretend this call cost you money: Why is the sky blue?"
    assert summary.splitlines() == [
        "prompt-cache: 1 hit, 1 miss out of 2 lookups (50.0% hit rate)",
        "  1 entry stored, 0.00 MB, 0 evictions, 1 call avoided",
    ]
    assert (tmp_path / ".demo-cache" / "cache.db").is_file()


def test_readme_quickstart_survives_a_new_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def build():
        @prompt_cache.cached(".demo-cache")
        def ask(prompt, model="gpt-4o"):
            calls.append(prompt)
            return "pretend this call cost you money: " + prompt

        return ask

    build()("Why is the sky blue?")
    again = build()
    assert again("Why is the sky blue?") == "pretend this call cost you money: Why is the sky blue?"
    assert calls == ["Why is the sky blue?"]  # the second decorator never called the function
    assert again.cache.stats().saved_calls == 1


def test_readme_manual_cache_block(tmp_path):
    from prompt_cache import Cache

    with Cache(tmp_path / "manual", ttl=86400, max_size_mb=200) as cache:
        answer = cache.get("Why is the sky blue?", model="gpt-4o")
        assert answer is None
        cache.set("Why is the sky blue?", "Rayleigh scattering.", model="gpt-4o")
        assert cache.get("Why is the sky blue?", model="gpt-4o") == "Rayleigh scattering."
