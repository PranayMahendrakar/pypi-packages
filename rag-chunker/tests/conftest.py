"""Shared sample documents."""
import pytest

PROSE = (
    "Solar panels turn light into power. Rooftop arrays are the common case. "
    "Inverters change the direct current into alternating current. "
    "Payback periods depend on local tariffs and on how much sun a roof gets. "
    "Cats sleep about sixteen hours a day. They hunt at dawn and at dusk. "
    "A kitten needs more sleep than an adult cat. "
    "Domestic cats keep the hunting instinct even when they are well fed. "
    "Bridges carry load through compression or through tension. "
    "A suspension bridge hangs its deck from cables. "
    "An arch bridge pushes the load outwards into the abutments."
)

MARKDOWN = """# Handbook

A short opening paragraph. It runs to two sentences.

## Installation

Install the package first. Then check the version.

```python
import example
example.run(retries=3)
print("done")
```

Some prose after the code block.

| step | command | note |
|------|---------|------|
| one  | build   | fast |
| two  | deploy  | slow |

## Troubleshooting

### Network

Check the proxy settings. Then retry the request.

### Disk

Free some space and run it again.
"""

HTML = """<html><head><title>ignored</title><style>p {color: red}</style></head>
<body>
<h1>Report</h1>
<p>The first paragraph has two sentences. Here is the second one.</p>
<h2>Details</h2>
<p>More prose lives here. It also has two sentences.</p>
<pre>def f(x):
    return x + 1</pre>
<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
<script>var ignored = 1;</script>
</body></html>"""

UNICODE = (
    "El informe seniala una subida del 12 por ciento. "
    "La reunion se celebro en Malaga con cafe y churros. "
    "Le rapport indique une hausse. L'equipe a confirme les chiffres. "
    "日本語の文章です。"
    "これは二番目の文。"
    "三番目もあります。"
    "التقرير يقول ذلك. "
    "Emoji survive too: \U0001f9ea \U0001f4da."
)

NO_PUNCTUATION = " ".join(["alpha beta gamma delta epsilon zeta eta theta"] * 30)


@pytest.fixture
def prose():
    return PROSE


@pytest.fixture
def markdown():
    return MARKDOWN


@pytest.fixture
def html():
    return HTML


@pytest.fixture
def unicode_text():
    return UNICODE


@pytest.fixture
def md_file(tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text(MARKDOWN, encoding="utf-8")
    return path


@pytest.fixture
def txt_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(UNICODE, encoding="utf-8")
    return path


@pytest.fixture
def html_file(tmp_path):
    path = tmp_path / "report.html"
    path.write_text(HTML, encoding="utf-8")
    return path
