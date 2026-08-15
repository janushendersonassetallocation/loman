# Design plan: serialization model and the `.loman` container

Status: **implemented**. This document is kept as the reasoning behind the
design; see [Saving computations](../user/features/other/saving_computations.md)
for how to use the result.

Where the implementation departed from the plan, or turned up something the plan
did not anticipate, it is recorded in [What changed during
implementation](#what-changed-during-implementation) at the end.

Three things are being solved, and the first two are prerequisites for the third:

1. **The value model is lossy and, for time series, broken.** A DataFrame with a
   `DatetimeIndex` cannot be serialized at all. Several types round-trip to
   something quietly different from what went in.
2. **The graph model drops node attributes** that the in-memory computation
   keeps — group, style, executor, converter, metadata, timing, and the values of
   `STALE` nodes.
3. **Everything is inline JSON.** A DataFrame is stored as nested lists of
   decimal numbers, so a modest frame produces a file several times its own
   in-memory size, slowly, with no compression available.

The answer to (3) is a container format, but a container over a value model that
cannot represent a timestamp is not worth building, which is why the ordering
below is not negotiable.

## Scope

**In scope**

- A lossless value model for the types loman users hold: datetimes, pandas
  indexes, numpy scalars, sets, bytes, non-finite floats, non-string dict keys.
- Full node fidelity across a round-trip.
- One logical on-disk layout, serialized either as a zip (`.loman`) or as a
  directory.
- Two explicit profiles — *readable* and *efficient* — over that one model.
- Out-of-line binary blobs with per-blob compression, and a pluggable way for
  user types to use them.
- `save`/`load`, with `write_json`/`read_json` kept and rewired.

**Out of scope**

- Removing `write_dill`/`read_dill`. They stay deprecated and untouched.
- Making the format safe to load from untrusted sources. It is not, it never was
  (`FunctionRefTransformer` imports whatever module a file names), and this plan
  documents that plainly and adds an opt-in `allow_code=False` rather than
  pretending otherwise.
- Incremental/delta writes, lazy loading, and mmap-backed arrays. The design
  keeps all three reachable; none is built here.
- A new serialization path for the UI widget or visualization layers.

## Decided, with evidence

These were measured against the current code, not reasoned about. They should
not need relitigating; the numbers are in the appendix.

**A time-series DataFrame cannot be saved today.** `DataFrameTransformer.to_dict`
does `transformer.to_dict(list(o.index))`, and no transformer handles
`pd.Timestamp`, so any `DatetimeIndex` raises `SerializationError`. The same
holds for a bare `Timestamp`, a `DatetimeIndex`, an `np.int64`, and a `set`. For
a library whose stated use case is post-mortem inspection of financial
pipelines, this is the headline defect.

**Inline JSON costs roughly 2.7× in size and 800× in time.** A 100k×10 float
frame occupying 8 MB in memory writes 21.5 MB of JSON in 0.81 s. `pickle` writes
8 MB in 0.001 s. The cause is `o.values.tolist()` and `o.ravel().tolist()`.

**Three types are silently corrupted.** `{1: 'a'}` returns as `{'1': 'a'}`; a
`MultiIndex` returns as a flat `Index` of tuples; DataFrame `columns` never pass
through the transformer at all. None raises; all three just hand back something
else.

**The files are not valid JSON.** `json.dump` defaults to `allow_nan=True`, so a
NaN or an infinity is written as a bare `NaN` / `Infinity` token. Python reads it
back; a strict parser — which is to say every non-Python consumer of a file whose
selling point is being readable — rejects it.

**Blanket compression is wrong in both directions.** On random float data,
DEFLATE bought 4% for 3.05 s. On realistic data (a rounded price series),
`npy` + `zlib:1` was **8× smaller in 0.01 s**. Neither always-on nor always-off
is defensible, so the choice has to be made per blob, from the data.

**Zip and directory containers cost the same to write and differ on update.**
Writing 128 MB took 0.05 s either way. Reading one value of eight took 0.011 s
from a stored zip and 0.002 s from a directory. Updating one value took 0.005 s
in a directory and a full container rewrite for the zip. That last row is the
whole argument for keeping the directory container first-class: it is exactly the
cost profile of checkpointing a long-running computation.

**Format version numbers cannot gate old readers.** `_from_dict` never reads
`data["version"]`. Every release to date ignores the field entirely, so
forward compatibility has to come from keeping changes additive. The version
field informs new readers; it cannot protect against old ones.

## The model

### One layout, two serializations

```text
manifest.json
blobs/0000.npy
blobs/0001.parquet
```

`.loman` is that tree inside a zip. A `dir` container is that tree on disk. One
specification, two ways of writing it down, and a test asserting the manifests
are byte-identical for the same input — so "one spec, not two" is enforced by CI
rather than by discipline.

**Blob filenames are never derived from node keys.** Zero-padded integer ids
sidestep `/` in hierarchical keys, Windows reserved names, case-insensitive
collisions, and unicode normalization in one move. Orientation for a human
poking around comes from a `node` field in the blob table.

### Manifest

```json
{
  "format": "loman-computation",
  "version": 3,
  "container": "zip",
  "profile": "efficient",
  "blobs": [
    {"id": 0, "path": "blobs/0000.npy", "codec": "npy",
     "compression": "zlib:1", "size": 8000128, "stored_size": 1004291,
     "node": "prices", "sha256": null}
  ],
  "metadata": {"": {}, "prices": {}},
  "nodes": [],
  "edges": []
}
```

`blobs` is an array indexed by id, so a reference is an integer and two nodes
holding the same object share one blob. `metadata` is keyed by node-key string
with `""` for the root, because `Computation._metadata` lives on the object
rather than on dag nodes. `sha256` is nullable and off by default — hashing
128 MB costs about 0.3 s for a guarantee most saves do not need.

A version 3 **single JSON document** is this same manifest with `"blobs": []` and
`"container": "json"`. There is one schema, serialized three ways.

### Blob references

A reference replaces the data payload and nothing else:

```json
{"type": "ndarray", "shape": [100000, 10], "dtype": "<f8",
 "encoding": "npy", "data": {"$blob": 0}}
```

This is the load-bearing decision of the whole design. Because the header stays
inline, the manifest still states the shape, dtype, column names and index type
of every value in the graph **without decoding a single byte** — readability
survives the efficient profile, which is what makes two profiles over one model
coherent rather than two formats wearing a trench coat.

It also means `Transformer.from_dict` needs exactly one new branch, and that
branch composes recursively wherever a value can appear: node values, constants
in `args`/`kwds`, list elements, attrs and dataclass fields. A value with no
useful inline header — a parquet frame, whose columns and dtypes live in the
footer — takes the same shape:
`{"type": "dataframe", "encoding": "parquet", "data": {"$blob": 3}}`.

**One hazard, fixed in the same change:** a user dict containing the literal key
`"$blob"` would be misread. `_dict_to_dict` already escapes a dict containing
`"type"`; that escape must generalize to any reserved key. It is the same line
that has to change for the non-string-key fix, which is why the two are done
together.

## API

```python
comp.save("run.loman")                      # efficient + zip (the default)
comp.save("run.loman", profile="readable")  # inline JSON, zipped
comp.save("run.json")                       # inferred: json container, readable
comp.save("run_dir", container="dir")
comp.save("run.loman", profile=EfficientProfile(compression="zstd:9"))

Computation.load("run.loman")               # container sniffed
Computation.load("legacy.json")             # v1 / v2 / v3 single document
```

**Profile and container are orthogonal axes.** Collapsing them into one setting
looks tidier and immediately fails on real cases: readable-inside-a-zip is
genuinely useful (a 21.5 MB manifest DEFLATEs to a few MB and stays a plain JSON
file any tool can open), and `dumps() -> str` can only ever be
readable-plus-single-document. Only one combination is invalid — efficient plus
`container="json"` — and it raises pointing at `container="zip"`, because
base64-inlining would inflate 33% and force a read-all.

Sniffing, in order: a directory, or a tree containing `manifest.json`, is `dir`;
a `PK\x03\x04` header is `zip`; a leading `{` is a single JSON document,
dispatched on `version` (absent, 1, 2, or 3); a dill pickle header produces an
error naming `read_dill`; anything else is a `SerializationError` listing the
three accepted forms.

`write_json` and `read_json` stay, undeprecated, as thin wrappers over
`save(container="json", profile="readable")` and `load()`. Their signatures and
the `serializer=` keyword are unchanged. They remain the right tool for a
diffable text artifact, which is a real thing to want and not a legacy path.

```python
@attrs.frozen
class SerializationProfile:
    name: str
    inline_max_bytes: int | None   # None = always inline; default 8 KiB
    compression: str               # "none" | "auto" | "zlib:1..9" | "zstd:N"
    array_encoding: str            # "json" | "npy"
    frame_encoding: str            # "json" | "npy" | "parquet"
    checksums: bool = False
    dedupe: str = "identity"       # "none" | "identity" | "content"
    overrides: Mapping[str, BlobSpec] = {}
```

`"readable"` and `"efficient"` resolve to prebuilt instances; an instance can be
passed for tuning. `ComputationSerializer` gains a `profile=` keyword, so it
composes with the existing `serializer=` hook rather than competing with it.

## Extension: a transformer is *offered* a blob sink

The requirement is that users can bring their own encoding for their own types.
Two obvious designs were rejected first.

**A type → codec registry, parallel to `Transformer`.** It would duplicate the
whole dispatch mechanism — `_direct_type_map`, `_subtype_order`, `order_classes`
— and create real ambiguity when a transformer and a codec both claim
`pd.DataFrame`. It is also unnecessary: on read, dispatch is already driven by
the `"type"` discriminator, so `codec` in the blob table is descriptive metadata
for third-party tooling, not a dispatch key.

**A `to_blob`/`from_blob` pair on `CustomTransformer`.** It gives a type two
serialization methods that must be kept in agreement, it is all-or-nothing per
type, and it cannot express "small readable header plus one large out-of-line
payload" — which is the shape nearly every real case wants.

What is proposed instead is one method, one code path, and a per-call decision:

```python
class NdArrayTransformer(CustomTransformer):
    def to_dict(self, t: Transformer, o) -> dict:
        head = {"shape": list(o.shape), "dtype": o.dtype.str}
        if t.offer_blob(nbytes=o.nbytes):
            head["encoding"] = "npy"
            head["data"] = t.put_blob(
                lambda f: np.save(f, o, allow_pickle=False),
                codec="npy", compressible=True,
            )
        else:
            head["encoding"] = "json"
            head["data"] = t.to_dict(o.ravel().tolist())
        return head
```

The properties that make this the right shape:

- **Nothing new to learn.** A user extends loman by writing a `CustomTransformer`
  exactly as today. `offer_blob` and `put_blob` are two optional calls on an
  object that was already being passed in.
- **No signature changes anywhere.** `to_dict(self, transformer, o)` is
  unchanged, so third-party transformers written against 0.6 keep working — they
  never call `offer_blob`, so they always inline, and lose nothing they had.
- **It composes, and composes partially.** A dataclass transformer inlines its
  scalar fields and offloads its array field, because `t.to_dict(field)`
  re-enters the same machinery.
- **It degrades correctly.** In the `json` container `offer_blob` returns `False`
  unconditionally, so no transformer ever needs to know which container it is
  writing into.

`put_blob` accepts `bytes | memoryview | Callable[[BinaryIO], None]`. The
callable form lets `np.save` or `pyarrow.parquet.write_table` stream directly
into the zip member via `ZipFile.open(name, "w")`, avoiding a full in-memory copy
of a large payload.

Write-time state — the sink, the profile, the node key being written — lives in a
`contextvars.ContextVar` scoped by `Transformer.writing(...)`. That avoids
signature churn and threading an object through user code, and it is thread- and
task-safe for free, which matters because loman has an executor model and someone
will eventually serialize from a worker. Outside the scope `offer_blob()` returns
`False` and `put_blob()` raises, so misuse is loud rather than silently inlining.

### Two levels of control, separated by what kind of fact they express

Per-save policy lives on the profile, as a selector-to-spec map:

```python
EfficientProfile(overrides={
    "market_data/**": BlobSpec(codec="parquet", compression="zstd:9"),
    "tag:already_compressed": BlobSpec(compression="none"),
})
```

Compression level is a property of *this file* — an archive versus a quick
checkpoint — not of the graph, and the same computation gets saved both ways.
Per-graph facts ("this node always holds already-compressed imagery") go in the
existing `add_node(metadata=...)` dict under a `loman.serialization` key.

**No `blob=` keyword is added to `add_node`.** It has thirteen parameters
already, and the fact belongs to the save, not to the node.

### Thresholds and compression

- **Out-of-line threshold**: `profile.inline_max_bytes`, default 8 KiB — about a
  1024-element float64 array. Below that, the per-member overhead plus a filename
  plus a seek costs more than the JSON, and keeping small values inline preserves
  the "open the manifest and read it" property.
- **Compression**: default `"auto"`, implemented as a sampled trial. Compress the
  first 256 KiB at `zlib:1`, extrapolate, and store raw if the projected saving
  is under 10%. The probe costs about a millisecond and resolves the
  4%-for-3-seconds versus 8×-for-10-milliseconds split directly from the data.
  `compressible=False` on `put_blob` skips the probe entirely for self-compressing
  payloads, which is how double-compression is prevented.
- **Blobs are always `ZIP_STORED`**, with compression applied by the blob layer
  before the bytes reach the container; `manifest.json` is always
  `ZIP_DEFLATED`. Beyond avoiding double-compression, stored members sit at known
  offsets, which keeps a future mmap or zero-copy read possible. DEFLATE members
  would foreclose that permanently.
- **Codecs**: arrays to `.npy` with `allow_pickle=False`; frames to parquet when
  pyarrow is importable, else an `.npy` block with columns, dtypes and index
  carried inline in the manifest. The fallback is a first-class tested path, not
  a degraded one — pyarrow does not install everywhere in the CI matrix.

## Phases

All of this lands as **one format version (3) and one release**. The phases are
commits, ordered so each is independently reviewable and revertable, and each
leaves the branch green.

**Phase 0 — hygiene.** `assert` to real exceptions in `Transformer.register_*`
(they vanish under `python -O`, turning duplicate registration into silent
overwrite). Delete `write_dill_old`, which `del`s `__getstate__`/`__setstate__`
off the class object and is therefore process-global and thread-unsafe. Fix the
`CustomTransformer(Point, to_dict=..., from_dict=...)` example in the user docs —
`CustomTransformer` is an ABC with no such `__init__`, so that snippet raises
`TypeError`. Add a doctest target over `docs/**/*.md` to CI, which is what let
that example ship in the first place. Deprecate `default_transformer`, which is
dead with respect to the computation path but is in `__all__`.
*Acceptance:* the doctest target passes and fails on a deliberately broken
example. *Depends on:* nothing. *Size:* small.

**Phase 1 — value-model correctness.** New transformers in
`serialization/values.py`: datetime, date, time, `Timedelta`, `Timestamp`,
`pd.Index` and `MultiIndex` as first-class, numpy scalars, `set` and `frozenset`,
`bytes`, `Decimal`. Non-string dict keys encoded losslessly. Non-finite floats
encoded as a tagged form **and then** `allow_nan=False` set, so invalid JSON
becomes structurally impossible — note that `allow_nan=False` on its own raises
rather than fixing anything. `columns` passed through the transformer. Reserved
key escaping generalized in `_dict_to_dict`. `STALE` values retained: delete
`_VALUE_STATES` and serialize whenever `NodeAttributes.VALUE` is present, letting
`has_value` carry the truth. In-memory `STALE` nodes already keep their values,
so the serializer is currently discarding information the model retains.
*Acceptance:* a tz-aware `DatetimeIndex` frame round-trips exactly; `{1: 'a'}`
returns with an int key; a `MultiIndex` returns as a `MultiIndex`; the output
parses under a strict JSON parser. *Depends on:* nothing. *Size:* large.

**Phase 2 — node fidelity.** Round-trip group, style, executor, converter,
metadata and timing — all currently hardcoded to `None` on load or never written.
Reconstruct ERROR exceptions from a whitelist of builtin types, with anything
else becoming `DeserializedError` carrying the original module, type name and
traceback; importing the module a file names would be
arbitrary-import-from-untrusted-data. Add `load(..., allow_code=False)` to refuse
`func_ref` and `dill_func` and load a values-and-structure-only graph, defaulting
to `True` to preserve current behaviour. The `dir` reader must reject blob paths
that are absolute or contain `..`; the zip reader is already safe because members
are read into memory and never extracted.
*Acceptance:* a computation with groups, styles, metadata and timing compares
equal after a round-trip; a `ValueError` node returns as a `ValueError`; a file
naming an arbitrary module loads under `allow_code=False` without importing it.
*Depends on:* phase 1 — `TimingData` holds datetimes. *Size:* medium.

**Phase 3 — the container.** `BlobStore` with zip, directory and inline
implementations; the manifest schema; `$blob` references;
`offer_blob`/`put_blob`/`get_blob`; the `contextvars` write scope;
`SerializationProfile`; `save` and `load`; sniffing; `write_json`/`read_json`
rewired. **One codec (`npy`), no compression** — the container is proved in
isolation before codecs are added on top.
*Acceptance:* the 100k×10 frame round-trips through all four
(profile, container) combinations; zip and dir produce byte-identical manifests;
a `.loman` opens with plain `zipfile`. *Depends on:* phases 1 and 2.
*Size:* large.

**Phase 4 — codecs and compression.** Parquet behind a new `loman[efficient]`
extra; the compression registry; the `auto` sampling heuristic; zstd; selector
overrides; identity dedup; optional content dedup and checksums.
*Acceptance:* the realistic price series lands under 1 MB; random floats are
stored raw rather than burning seconds; the npy fallback has line coverage on a
job with no pyarrow. *Depends on:* phase 3. *Size:* medium.

**Phase 5 — documentation.** Rewrite the format reference in
`serializing_computations.md`, the migration guide, the marimo notebook, the
mkdocs nav and the changelog. Fix the stale enum reprs in the user docs — the
code has `ERROR = 5, PINNED = 6` and the docs print `<States.PINNED: 5>` and
`<States.ERROR: 4>` — which the phase 0 doctest target will force anyway.
*Depends on:* the rest. *Size:* medium.

Phases 0 to 2 are worth having even if the container work is cut short: they turn
a serializer that cannot save a time series into one that can, and they do not
depend on any of the container design.

## Testing approach

**Restructure first, as a no-op commit.** Split the 1799-line
`tests/test_serialization.py` into `tests/serialization/` — transformer, values,
blobs, profiles, containers, format compat, computation round-trip. Move tests
verbatim so the commit reviews as pure relocation. Do the phase 0 assertion
change *before* the split, so it does not tangle with the duplicate-registration
tests.

**One matrix catches most of it.** About forty awkward values crossed with the
four valid (profile, container) combinations: NaN, ±inf, `{1: 'a'}`,
`{'type': 'x'}`, `{'$blob': 1}`, a tz-aware `DatetimeIndex` frame, a `MultiIndex`
frame, an empty frame, an object-dtype column, a 0-d array, `np.float32` and
`np.int64` scalars, `set`, `bytes`, a nested `Computation`, an ERROR node, a
STALE node, and a node with constants. A curated list beats property-based
testing here, and `hypothesis` is not installed.

**Container invariants**, as assertions rather than conventions:

- `manifest.json` parses under `json.loads(..., parse_constant=_raise)`, which
  mechanically proves the non-finite-float fix stays fixed.
- Every `$blob` id resolves; no orphans, no duplicate ids.
- Zip and dir produce byte-identical manifests for the same input.
- Two saves of the same computation produce identical bytes. This needs
  `ZipInfo.date_time` set explicitly, since Python defaults it to *now*, and
  members sorted. It is worth the trouble: it enables content-addressed caching
  and makes saved graphs diffable.

**Golden files.** Small fixtures written by format versions 1, 2 and 3, asserted
to still load. Given that `_from_dict` ignores the version field, this is the
only real compatibility defence there is. Keep them small — CI uses git LFS and
these must stay well away from it.

**Optional dependencies under the coverage gate.** `importorskip("pyarrow")` for
the parquet path, *plus* a test that monkeypatches `loman._extras.require` to
raise, forcing the npy fallback. Otherwise the fallback has no coverage on any
job where pyarrow is absent, which is where it actually runs.

**A performance guard**, marked `@pytest.mark.stress` to match the existing
marker and the `tests/stress/` precedent: the 100k×10 frame writes in under
roughly 0.2 s and 10 MB. Deselected from normal runs; it exists to catch a silent
regression back to `tolist()`.

## Caveats

- **The format is not safe to load from untrusted input**, and this plan does not
  make it so. `func_ref` imports whatever module a file names and `dill_func`
  executes a pickle. `allow_code=False` is a mitigation for people who know they
  need it, not a security boundary, and the docs should say exactly that.
- **Updating one value in a `.loman` rewrites the whole container**, and the cost
  scales with total size. Anyone checkpointing a multi-gigabyte computation in a
  loop should use `container="dir"`, which is why it is documented and tested
  rather than hidden.
- **Old readers ignore the version field**, so a version 3 file opened by a 0.6
  release will not be rejected — it will be misread. Nothing in this plan can fix
  that retroactively; it is an argument for keeping every future change additive.
- **`use_dill_for_functions` remains as non-portable as it ever was.** Nothing
  here improves it, and the efficient profile does not make a dill blob any more
  durable.
- **Two write APIs will coexist.** `save`/`load` deliberately breaks the
  `write_<format>` convention, because the point of `save` is that you do not
  pick a format. `write_json` stays because a diffable text artifact is a
  legitimate thing to ask for by name.

## Open questions

- **Should the efficient profile checksum by default?** It is about 0.3 s per
  128 MB. Off is proposed, on the grounds that most saves are checkpoints and
  the zip already carries CRC32s.
- **Content dedup, or identity dedup only?** Identity (`id()`-based) is free and
  catches the common case of one object on two nodes. Content dedup needs a full
  hash of every blob and only pays off across repeated saves, which is really a
  delta-write feature.
- **What does `dumps()` do under a non-json container?** Currently proposed to
  raise. Returning base64 of the zip is possible and probably a trap.
- **Should `load` accept a URL or an object-store path?** Out of scope here, but
  the blob table makes ranged reads from S3 tractable later, and that is worth
  not designing out.
- **Does the typed-computations work interact?** `_from_dict` hardcodes
  `Computation()`, so a subclass is lost on load — noted as a caveat in
  `TYPED_COMPUTATIONS_PLAN.md`, and the manifest is the natural place to record
  the class if that is ever wanted.

## Evidence appendix

All figures from the current code on this branch.

**Inline JSON, 100k×10 float frame (8 MB in memory)**

| | size | write time |
| --- | --- | --- |
| `write_json` | 21.5 MB | 0.81 s |
| gzip of that | 9.6 MB | +0.87 s |
| `pickle` | 8.0 MB | 0.001 s |
| `.npy` | 8.0 MB | 0.001 s |

**Value round-trip, current behaviour**

| value | result |
| --- | --- |
| frame with `DatetimeIndex` | `SerializationError` |
| `pd.Timestamp`, `DatetimeIndex` | `SerializationError` |
| `np.int64` scalar, `set` | `SerializationError` |
| `{1: 'a'}` | returns `{'1': 'a'}` |
| `MultiIndex` frame | returns a flat `Index` of tuples |
| array with NaN/inf | returns correctly, file is invalid JSON |

**Containers, 128 MB across 8 nodes**

| | write | read 1 of 8 | update 1 value |
| --- | --- | --- | --- |
| zip STORED | 0.05 s | 0.011 s | full rewrite, 0.074 s |
| zip DEFLATE | 3.05 s | 0.044 s | full rewrite |
| directory | 0.05 s | 0.002 s | 0.005 s |
| directory + mmap | — | 0.006 s, lazy | — |

**Compression, 500k-row realistic price series (`.npy`, 4 MB)**

| | size | time |
| --- | --- | --- |
| raw | 4.00 MB | — |
| `zlib:1` | 0.50 MB | 0.01 s |
| `zlib:6` | 0.36 MB | 0.09 s |
| `lzma:1` | 0.31 MB | 0.17 s |

Against random floats the same codecs bought roughly 4%. The gap between these
two tables is the entire argument for sampling.

## What changed during implementation

Five things the plan did not anticipate, and one it got wrong.

**Exact-type dispatch, not `isinstance`.** `Transformer.to_dict` opened with
`isinstance(o, (int, float))`, and `np.float64` *is* a `float` while an `IntEnum`
member *is* an `int` — so both were returned as bare numbers with their type
silently discarded. `np.int64` was not, which is why only that one raised. The
fast path now checks `type(o) is float` and friends, and anything else falls
through to the transformers. This was a prerequisite for the numpy scalar work
rather than a separate fix.

**Pandas resolution is not nanoseconds.** `DatetimeIndex.asi8` returns ticks in
the index's *own* unit, which is microseconds by default in pandas 3. The first
implementation assumed nanoseconds and read every timestamp back as 1970. The
unit is now recorded alongside the values, for indexes, `Timestamp` and
`Timedelta` alike.

**Indexes needed the blob path too.** The plan treated out-of-line storage as a
question about values. But a 100k-row frame's `DatetimeIndex` is 100k integers,
and leaving it inline kept the manifest at roughly 800 KB — defeating the
"readable manifest" property the design rests on. Indexes and frame columns now
delegate to the ndarray transformer, so they inherit blob storage. The manifest
for that frame is about 3 KB.

**Parquet is not automatically smaller.** The plan assumed parquet would be the
better frame encoding. Measured on a realistic 200k-row frame, `.npy` plus
sampled zlib came out at 1.03 MB against parquet-with-zstd at 1.45 MB: rounding
creates repetition that zlib exploits directly. Parquet stayed as an opt-in
`frame_encoding="parquet"` for cross-tool readability, and `.npy` is the default.

**The `assert` on duplicate registration had no test.** The plan listed replacing
it as hygiene. It was, but note that nothing broke when it changed, which is the
point: under `python -O` those asserts were absent and a duplicate registration
silently overwrote the earlier transformer.

### Doctests found more than expected

The plan added a doctest target expecting it to catch the one wrong
`CustomTransformer` example. It caught twelve of thirteen documentation pages,
including the *first* example in the serialization page — `to_dict()` returns
`NodeKey`-keyed dicts, not string-keyed ones. Nothing had ever executed them.

Two consequences worth recording:

- The pages this work owns are fixed and enforced; the rest are listed in
  `KNOWN_STALE_DOCS` in `tests/test_docs.py`, with a test that fails if one
  starts passing, so the list only shrinks. Their common cause is a first block
  that uses names it never imports.
- A page printing a `set` is order-dependent under hash randomisation, so it
  passes in some processes and fails in others. Such a page must print a sorted
  form before it can be enforced, or it makes the guard flaky rather than
  useful. `tagging_nodes.md` was fixed this way.

## Second round: pluggable storage and pandas guarantees

Three questions after the first round, and what each turned up.

**"Can a user bring their own serialisation?"** Half. *Encoding* was pluggable
via `CustomTransformer` and `offer_blob`/`put_blob`; *destination* was not. The
container was a hardcoded dispatch to zip-or-directory and blobs always landed
inside it.

Fixed by splitting the two roles that had been conflated in one class.
`BlobStore` is now only *where bytes go* --- `write_blob`, `read_blob`, and an
optional `key_for` --- while `BlobWriter` and `BlobReader` own the ids,
compression, deduplication, checksums and blob table. A store for S3 or a
database is therefore two methods with no loman internals in them, which
`tests/test_byos.py` asserts by implementing one exactly that way.

Routing is a node default that a save can override: `add_node(store='warehouse')`
states what kind of thing a node is, and a profile override states where that
kind goes today, so the same computation reaches a bucket in production and a
plain container in a test.

Two decisions inside that are worth keeping:

- **A saved file records a store's name, never its configuration.** No bucket, no
  credential ever reaches the file. The cost is that a file cannot resolve its
  own external values --- whoever loads it supplies the store --- and that cost
  was taken deliberately.
- **A missing store is an error, not a fallback.** The first implementation
  silently inlined when a node's store was absent, which would put data in the
  file while the caller believed it had gone to their bucket. A test caught it.

**Also fixed: `tag:` selectors matched nothing.** `settings_for` accepted tags;
the only caller never passed them. It had a unit test --- against the function,
not the behaviour --- which is why it looked covered. The lesson is the same one
the doctest work taught: test the path users take.

**"Long-term storage versus quick debugging?"** Both, but the documentation
still carried a blanket "not intended for long-term storage" note that the
versioned format had outgrown. Replaced with the split that is actually true:
values are durable and version-guarded; a node's *function* is only as durable as
the module path it names, and a dill-serialized one is not portable across Python
versions. A saved computation is a durable record of what was computed and a
best-effort record of how.

**"Is it compatible with pandas 2 and 3?"** It was, accidentally. The full
serialization suite passed unchanged on pandas 2.3.3, and files interoperated
both directions --- because the unit-recording fix from the first round means a
pandas-2 `ns` index stays `ns` under pandas 3 rather than being reinterpreted.

But `pyproject.toml` declared `pandas >= 0.19.2`, which was fiction: `.as_unit`
and `DatetimeIndex.unit` arrived in 2.0, and below that resolution would be
silently wrong rather than an import error. The floor is now `>=2.0`, with a test
asserting it and listing the APIs that set it. A committed pandas-2 fixture
guards the old-file direction in a single environment, and a repo-owned CI
workflow runs the suite against the minimum supported pandas --- repo-owned
because `.github/workflows/rhiza_ci.yml` is in `template.lock` and `make sync`
would overwrite an edit to it.

### Still open

- **`write_dill` / `read_dill`** remain deprecated and untouched, as scoped.
- **Delta writes, lazy loading and mmap** are still deferred. Blob references
  make all three reachable; none is built.
- **`RUF036` in `computeengine.py`** (two `None`-not-last type unions) predates
  this work and is untouched.
- **A store cannot stream.** `BlobWriter` materialises each payload as bytes to
  compress and hash it, so a value larger than memory cannot be written. Fixing
  that means a streaming path that skips compression and content dedup.
