# Saving computations

`save` and `load` write a computation to disk and read it back. The default is a
`.loman` file: a single archive holding a `manifest.json` that describes the
graph, plus large values stored beside it as binary.

```pycon
>>> import numpy as np, pandas as pd
>>> from loman import Computation
>>> prices = pd.DataFrame(
...     {"px": np.arange(100_000, dtype="float64")},
...     index=pd.date_range("2020-01-01", periods=100_000, freq="min"),
... )
>>> comp = Computation()
>>> comp.add_node("prices", value=prices)
>>> comp.save("run.loman")
>>> comp2 = Computation.load("run.loman")
>>> comp2.v.prices.equals(prices)
True
```

## Profiles and containers

Two independent choices. The **profile** decides how a value is encoded; the
**container** decides where the bytes land.

| | `json` | `zip` (`.loman`) | `dir` |
|---|---|---|---|
| `readable` | one JSON document | zipped JSON | JSON in a folder |
| `efficient` | *not possible* | **the default** | folder + binary blobs |

```pycon
>>> comp.save("run.loman")  # efficient + zip (the default)
>>> comp.save("run.loman", profile="readable")  # inline JSON, still zipped
>>> comp.save("run.json")  # single JSON document
>>> comp.save("run_dir", container="dir")  # same layout, unzipped
```

The container is inferred from the path — `.json` means a single document,
anything else means a `.loman` archive — and `load` detects it from the file
itself, so you never have to say which you have.

The one combination that cannot work is `efficient` with a single JSON document,
since there is nowhere to put the out-of-line bytes. It raises and says so.

### The efficient profile is still inspectable

Only the bulk data moves out of line. Shape, dtype, column names and index type
stay in the manifest, so you can see what a saved graph contains without
decoding any of it:

```pycon
>>> import json, zipfile
>>> comp.save("run.loman")  # the default: efficient
>>> manifest = json.loads(zipfile.ZipFile("run.loman").read("manifest.json"))
>>> value = manifest["nodes"][0]["value"]
>>> value["index"]["kind"], value["index"]["freq"]
('datetime', 'min')
>>> manifest["blobs"][0]["codec"]
'npy'
```

For the 100k-row frame above, the manifest is a couple of kilobytes whatever the
data weighs.

### Which container to use

`.loman` is one file, which is what you want for handing a run to someone else or
uploading it somewhere.

Use `container='dir'` when you save **repeatedly** — a checkpointing loop, say.
Updating one value in a zip means rewriting the whole archive, at a cost that
grows with its size; a directory rewrites only the file that changed.

## Compression

Blobs are compressed with **zstd at level 1** by default. You choose the codec;
nothing is inferred from the data.

```pycon
>>> comp2 = Computation()
>>> comp2.add_node("rounded", value=np.round(np.arange(200_000) * 0.01, 2))
>>> comp2.save("rounded.loman")
>>> entry = json.loads(zipfile.ZipFile("rounded.loman").read("manifest.json"))["blobs"][0]
>>> entry["compression"]
'zstd:1'
>>> entry["stored_size"] < entry["size"] // 5
True
```

To choose something else, pass a profile:

```pycon
>>> from loman import SerializationProfile
>>> archive = SerializationProfile("archive", inline_max_bytes=8192, compression="zstd:19")
>>> comp.save("small.loman", profile=archive)
>>> fast = SerializationProfile("fast", inline_max_bytes=8192, compression="none")
>>> comp.save("fast.loman", profile=fast)
```

`compression` accepts `"none"`, or a codec and optional level: `"zstd:1"` to
`"zstd:22"`, `"zlib:1"` to `"zlib:9"`. You can register your own with
`loman.serialization.compression.register_codec`.

### Why zstd, and why on by default

Compressing is worth doing without being asked only if finding out costs almost
nothing. Measured on 4 MB payloads:

| | reject incompressible | realistic price series |
|---|---|---|
| zlib:1 | 43 MB/s | 8.0× at 291 MB/s |
| **zstd:1** | **1067 MB/s** | **9.3× at 568 MB/s** |

zstd rejects data it cannot compress about 25 times faster than zlib, and
compresses real data better and faster besides. That is why it is a required
dependency rather than an optional one: it turns "compress and see" from a
decision into a non-event, at roughly a second per gigabyte of incompressible
data.

### What gets stored

Whichever is smaller. After compressing, the two sizes are compared and the
smaller one is written; a blob that did not shrink is stored exactly as it came
and recorded as `"none"`:

```pycon
>>> import os
>>> comp3 = Computation()
>>> comp3.add_node("noise", value=np.frombuffer(os.urandom(500_000), dtype=np.uint8))
>>> comp3.save("noise.loman")
>>> json.loads(zipfile.ZipFile("noise.loman").read("manifest.json"))["blobs"][0]["compression"]
'none'
```

That is a byte comparison on data already in hand, not an estimate, so there is
no threshold to tune. Without it an incompressible blob would be stored slightly
*larger* than raw — codecs add framing — and every future read would pay a
decompression step for nothing.

!!! note
    An earlier version of this had an `"auto"` mode that compressed the first
    256 KiB of each blob and extrapolated. It was removed rather than tuned: on
    a payload whose character changes part way through — common in market data —
    it was wrong in both directions, once projecting a 3.6% saving against an
    actual 36.8% and storing the blob raw. With zstd there was nothing left for
    it to save.

## Optional extras

Everything above works with loman's own dependencies: values are stored as
`.npy` and compressed with zstd, both of which come with loman.

```bash
pip install 'loman[efficient]'
```

adds `pyarrow`, for storing DataFrames as parquet:

```pycon
>>> parquet = SerializationProfile("pq", inline_max_bytes=8192, frame_encoding="parquet")
>>> comp.save("run_pq.loman", profile=parquet)
```

Parquet's value is that other tools can read the blobs directly. It is not
automatically smaller: on a 200k-row price frame the default `.npy`-plus-zstd
path came out at 1.21 MB against parquet's 1.46 MB, because zstd exploits the
repetition that rounding creates. Measure before assuming. If pyarrow cannot represent a particular frame (duplicate column
names, for instance), the save falls back to the default encoding rather than
failing.

## Bring your own storage

Sometimes a value does not belong in the saved file at all — a frame that should
land in a bucket as parquet, or a result that belongs in a warehouse table. Mark
the node with a store name, and supply the implementation when you save and load:

```python
comp.add_node("prices", value=frame, store="warehouse")

comp.save("run.loman", stores={"warehouse": S3Store(bucket="my-bucket")})
comp2 = Computation.load("run.loman", stores={"warehouse": S3Store(bucket="my-bucket")})
```

The archive then holds only the manifest; `prices` lives in your bucket, and the
manifest records which store holds it so `load` knows to ask.

### Writing a store

A store is two methods. Compression, deduplication, checksums, blob ids and the
blob table are all handled for you:

```pycon
>>> from loman.serialization import BlobStore
>>> class DictStore(BlobStore):
...     def __init__(self):
...         self.blobs = {}
...
...     def write_blob(self, key, data):
...         self.blobs[key] = data
...
...     def read_blob(self, key):
...         return self.blobs[key]
```

```pycon
>>> store = DictStore()
>>> comp5 = Computation()
>>> comp5.add_node("prices", value=prices, store="warehouse")
>>> comp5.save("remote.loman", stores={"warehouse": store})
>>> len(store.blobs) > 0
True
>>> zipfile.ZipFile("remote.loman").namelist()
['manifest.json']
>>> Computation.load("remote.loman", stores={"warehouse": store}).v.prices.equals(prices)
True
```

`key` is a short relative path such as `blobs/0000.npy`. A real store would put
it under a prefix of its own; it just has to return the same bytes for the same
key. Override `key_for(blob_id, codec, node)` to lay keys out differently — by
node name, or partitioned by date.

### The file never holds your credentials

A manifest records a store's **name**, never its configuration. No bucket, no
connection string, no key ever reaches the file:

```pycon
>>> manifest = json.loads(zipfile.ZipFile("remote.loman").read("manifest.json"))
>>> manifest["blobs"][0]["store"]
'warehouse'
```

The consequence is that a file with external values cannot resolve them on its
own — whoever loads it supplies the matching store. If they don't, the error says
which store is missing and for which node, rather than silently returning a
half-loaded graph.

The same applies at save time: a node routed to a store you forgot to pass is an
error, not a quiet fallback to writing the data into the file. Believing your
data went to S3 when it is sitting in the archive is a worse outcome than a
failed save.

### Routing at save time instead

A node's `store=` is a default, not a fixture. A profile override wins, so the
same computation can go to a bucket in production and to a plain container in a
test:

```pycon
>>> local = SerializationProfile("local", inline_max_bytes=8192, overrides={"prices": {"store": None}})
>>> comp5.save("local.loman", profile=local)
>>> "blobs/0000.npy" in zipfile.ZipFile("local.loman").namelist()
True
```

Selectors match node-key globs (`'market/*'`) or tags (`'tag:bulky'`), so you can
tag nodes by *what they are* and let each save decide where that kind of thing
goes.

### A readable manifest with the data elsewhere

Because an external store is independent of the container, you can combine a
plain JSON manifest with out-of-line values:

```python
comp.save("run.json", profile=efficient, stores={"warehouse": S3Store(...)})
```

The result is a manifest you can read in a text editor, describing values that
live in your bucket.

## Deduplication

Two nodes holding the same object store one blob:

```pycon
>>> shared = np.arange(50_000, dtype="float64")
>>> comp3 = Computation()
>>> comp3.add_node("a", value=shared)
>>> comp3.add_node("b", value=shared)
>>> comp3.save("shared.loman")
>>> len(json.loads(zipfile.ZipFile("shared.loman").read("manifest.json"))["blobs"])
1
```

That is identity-based, so two *equal but separate* arrays are still stored
twice. `SerializationProfile(..., dedupe='content')` hashes instead, catching
those at the cost of digesting every blob.

## Loading untrusted files

Loading restores node functions, which means importing the modules the file names
— or unpickling a dill blob out of it. **Both run code the file chose.** The
format is not safe against a hostile file, and never was.

`allow_code=False` skips resolving callables entirely. Values, structure, states
and tags still load, so the graph can be inspected but not recalculated:

```pycon
>>> comp4 = Computation.load("run.loman", allow_code=False)
>>> comp4.v.prices.shape
(100000, 1)
```

This is a mitigation for people who know they need it, not a security boundary.
Prefer not loading files you do not trust.

## File layout

Both containers hold the same tree. `.loman` is it zipped; `dir` is it on disk:

```text
manifest.json
blobs/0000.npy
blobs/0001.npy
```

Blob filenames are integers, never node names, which sidesteps `/` in
hierarchical keys, Windows reserved names and case-insensitive collisions. The
blob table records which node each one came from.

```json
{
  "version": 3,
  "container": "zip",
  "profile": "efficient",
  "blobs": [
    {"id": 0, "path": "blobs/0000.npy", "codec": "npy",
     "compression": "zlib:1", "size": 800128, "stored_size": 98241,
     "node": "prices"}
  ],
  "nodes": [ ... ],
  "edges": [ ... ],
  "metadata": { ... }
}
```

A value stored out of line keeps its descriptive header inline and replaces only
its data:

```json
{"type": "ndarray", "shape": [100000], "dtype": "<f8",
 "encoding": "npy", "data": {"$blob": 0}}
```

The single-document container is the same manifest with `"blobs": []`, which is
why one `load` reads all three.

## Relationship to `write_json`

`write_json` and `read_json` still work, unchanged and undeprecated. They are
`save`/`load` restricted to the readable single-document container, and remain
the right choice when you want a text file you can diff.

See [Serializing computations](serializing_computations.md) for the value
encoding, custom types, and the `serialize=False` flag — all of which apply to
`save` too.
