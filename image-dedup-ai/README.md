# image-dedup-ai

Find duplicate and near-duplicate images across a large folder, and keep the hashes in an index file so the next scan only reads new or edited files. Matching uses perceptual hashes, which are a heuristic: they find resized and re-compressed copies, but not rotated, flipped or heavily cropped ones (an `embed=` hook covers those).

## Install

```
pip install image-dedup-ai
```

Needs only numpy and Pillow. No OpenCV, no torch, no model downloads, no network access.

## Quickstart

```python
import image_dedup_ai
from PIL import Image

photo = Image.effect_mandelbrot((512, 512), (-2.0, -1.5, 1.0, 1.5), 100)
images = {"photo": photo, "photo_half_size": photo.resize((256, 256)), "gradient": Image.linear_gradient("L")}
result = image_dedup_ai.find_duplicates(images)   # or a folder: find_duplicates("D:/Photos")
print(result.summary())                            # 1 group: keep "photo", "photo_half_size" is a copy
```

## What it does

- **Hashes each image once.** Folders are scanned recursively (links to other folders are not followed). Each
  file is read and decoded once for hashing (large JPEGs at reduced size), turned into a small greyscale square,
  and three perceptual hashes are stored in the index, a SQLite file (or memory when no path is given):
  - `phash` (default): the DCT (computed with numpy) of a 32x32 downscale, low frequencies compared with their
    median. Robust to resizing, re-compression and contrast changes.
  - `dhash`: whether each pixel is brighter than its neighbour. Fast, but it calls more look-alike images
    duplicates than `phash`; use a stricter `threshold` such as 0.95.
  - `ahash`: whether each cell is brighter than the mean. The fastest and the weakest.
- **Similarity** is `1 - hamming / bits` (64 bits with the default `hash_size=8`). Images at or above `threshold`
  are linked, and the connected sets of links are the duplicate groups.
- **Re-checks instantly.** Each entry is keyed on the file's path, size and modification time. Adding the same
  folder again skips unchanged files (and files already known to be unreadable), rehashes edited ones and
  drops entries whose files are gone.
- **Scales.** Large indexes are searched with multi-index hashing: the hash is split into chunks so only images
  that agree exactly on some chunk are compared. The result is identical to comparing every pair.
  In testing, grouping 100,000 indexed images (well-spread hashes, default threshold) took under a second;
  collections full of look-alike images take longer.
- **Keeps one per group**: the largest resolution, then the largest file, then the shortest path (so
  `photo.jpg` is kept over `photo - Copy.jpg`). `wasted_bytes` is the size of all the other members.
- **Never crashes a scan on a bad file.** Unreadable, truncated, empty or non-image files are skipped with a
  reason and reported in `result.skipped`. Inside a folder, only image extensions are read; other files are
  counted, not opened. EXIF orientation is honoured; transparency is flattened onto white; 16-bit images are
  scaled, not clipped.
- **Never modifies anything.** Files are opened read-only and PIL images passed in are left exactly as they were.
  Nothing is deleted; `result.drop` is a list for you to act on.

### What perceptual hashing cannot catch

- **Rotated or flipped copies** are not found: a mirrored photo has a very different hash. This is the honest
  limit of perceptual hashing, and the tests assert it rather than hide it.
- **Crops of more than a small margin** (in the tests a 2% trim per edge is still found and a 10% trim is not), and heavy edits
  such as strong brightening that clips highlights, filters, overlays or collages, change the hash too much.
- **Near-blank images** (solid colours, empty pages, black frames) carry no detail, so their hash is noise. They
  are grouped only when their content is identical, never by hash.
- **Different photos of the same scene** (burst shots, a second take) usually differ by more than a copy does.

To catch those, pass an embedding: `Index(path, embed=fn)` calls `fn(image)` with an upright RGB `PIL.Image`
for every added image and stores the vector it returns. `find_duplicates(method="embed")` then groups by cosine
similarity. Any model works (CLIP, a colour histogram, your own); this package does not ship or download one.
The embedding search compares every pair, which is practical up to a few tens of thousands of images.

## API

```python
image_dedup_ai.find_duplicates(folder, *, threshold=0.9, method="phash", hash_size=8, recursive=True,
                               index=None, embed=None, workers=None) -> DedupeResult
```

`folder` is anything `Index.add` accepts. `index` is an optional file path that keeps the hashes between calls.

For a real folder, keep the index so the second run only reads new or edited files:

```python
with image_dedup_ai.Index("photos.idx") as idx:
    idx.add("D:/Photos")
    result = idx.find_duplicates(threshold=0.9)
print(result.summary())   # result.keep, result.drop and result.wasted_bytes say what to do next
```

`Index(path=None, *, hash_size=8, embed=None)`: `path` is the SQLite file (`None` keeps everything in
memory). `hash_size` is 4 to 32 (`hash_size ** 2` bits). A file is tied to its `hash_size`; opening it with
another raises a `ValueError` that says which to use.
Supports `with Index(...) as idx:`.

- `.add(images, *, recursive=True, workers=None) -> int`: a folder, an image path, a `PIL.Image`, a
  `{name: PIL.Image}` dict, or a list mixing them. Returns how many of them are now indexed. Unnamed PIL
  images are named `<image 1>`, `<image 2>`, ... `workers` is the number of decoding threads (default up to 8).
  `idx.last_add` is an `AddReport` with `hashed`, `unchanged`, `embedded`, `removed`, `skipped`
  (list of `(path, reason)`), `ignored` (non-image files) and `.summary()`.
- `.find_duplicates(*, threshold=0.9, method="phash") -> DedupeResult`: `method` is `"phash"`, `"dhash"`,
  `"ahash"` or `"embed"`. `threshold` is in `(0, 1]`; `1.0` means identical hashes only.
- `.near(image, *, k=5, method="phash") -> list[Match(path, similarity)]`: the `k` most similar indexed images to
  a path or `PIL.Image`, most similar first. The query is not added, and is left out of its own results.
- `.remove(path) -> int`: remove one entry, or every entry under a folder. `.count()`, `len(idx)`, `.close()`.

`image_dedup_ai.hash_image(image, method="phash", *, hash_size=8) -> str` returns one hash as hex.

`DedupeResult`

- `.groups` - `list[list[str]]`, each group's paths with the kept image first; largest reclaimable size first
- `.keep` - `list[str]`, the kept path of each group; `.drop` - every other member (the copies)
- `.pairs` - `list[(a, b, similarity)]`, the links that formed the groups (`a < b`). Byte-identical files, and
  files with identical hashes, are each linked once to the best copy among them rather than to every other
  member, so the list grows linearly with the number of copies.
- `.details` - per group, a `Member(path, width, height, bytes, similarity, identical, kept)` for each image,
  where `similarity` and `identical` are relative to the kept image
- `.wasted_bytes`, `.n_groups`, `.n_duplicates`, `.n_images`, `.method`, `.threshold`, `.bits`
- `.skipped` - `list[(path, reason)]`; `.notes` - caveats that apply to this result
- `.summary(max_groups=10)` - plain-text report; `.to_dict()` - JSON-safe dict

## CLI

```
image-dedup-ai PATH [PATH ...] [--index FILE] [--method phash|dhash|ahash] [--threshold 0.9]
               [--hash-size 8] [--no-recursive] [--workers N] [--near IMAGE] [-k 5]
               [--json] [--output FILE]
```

- `image-dedup-ai D:/Photos` prints the summary.
- `image-dedup-ai D:/Photos --index photos.idx --output dupes.csv` keeps the hashes and writes one CSV row per
  image in a group (`group, action, path, width, height, bytes, similarity_to_keep, identical_to_keep`);
  `--output dupes.json` writes the full result.
- `image-dedup-ai --index photos.idx --near holiday.jpg -k 10` lists the nearest indexed images.
- `--json` prints `to_dict()` (plus a `scan` section) as JSON. Output is UTF-8, so non-ASCII file names are safe
  in pipes and redirects.

Nothing is ever deleted or moved by the CLI.

## License

MIT
