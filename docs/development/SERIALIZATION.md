# Serialization and storage: design notes

Developer-facing notes on how loman's serialization is put together, why the
scope is drawn where it is, and what is planned next. For usage, see
[Serializing computations](../user/features/other/serializing_computations.md).

## Scope: the artifact, not the storage

Loman's job is to produce a **compact, faithful, durable artifact**. Where that
artifact is put is the caller's business.

This is a deliberate boundary. Loman's read and write methods accept file-like
objects as well as paths, so remote storage is already one line away:

```python
comp.write_archive(fsspec.open("s3://bucket/run.loman", "wb").open())
```

Owning storage backends would mean owning credentials, retries, partial writes,
and a matrix of optional dependencies, in exchange for something the ecosystem
already does better. Run registries, retention policies, cross-run
deduplication and indexing sit on the same side of the line.

What loman does owe its users, and now commits to, is that the artifact is
small, exact, and readable by future versions.

## Layout

| Module | Responsibility |
|---|---|
| `serialization/computation.py` | Graph → manifest dict; version negotiation |
| `serialization/transformer.py` | Type dispatch registry and per-type transformers |
| `serialization/arraycodec.py` | Dtype-faithful 1-D array and index encoding |
| `serialization/archive.py` | The zip container and its payload encodings |

Two containers share one schema. `write_json` emits the manifest with every
value inline; `write_archive` emits the same manifest with large values
replaced by payload references, alongside the payload entries. Because there is
one schema there is one version counter, one compatibility policy, and one
golden corpus.

`ComputationSerializer` exposes `_encode_value` / `_decode_value` as the single
hook the archive overrides. Anything that needs to intercept values should go
through those rather than duplicating the node walk.

## Format versioning

`FORMAT_VERSION` is what we write; `MIN_SUPPORTED_VERSION` is the oldest we
read. The promise is that a reader accepts everything in between.

When changing the schema:

1. Bump `FORMAT_VERSION`.
2. Run `uv run python scripts/generate_format_goldens.py` to capture a corpus
   for the new version. It refuses to overwrite an existing version's
   directory — regenerating an old corpus with new code would destroy the only
   evidence of what that version actually looked like.
3. Leave the old decode path in place. `DataFrameTransformer._from_dict_v1` is
   the pattern: value encodings are self-describing, so a decoder branches on
   the shape of the payload rather than on the document version.

`tests/format_fixtures.py` holds the functions golden files reference by import
path. Those names are load-bearing — renaming one breaks every corpus that
mentions it.

## Why column-wise encoding

Format version 1 encoded frames via `DataFrame.values.tolist()`. That routed
everything through `object` whenever a frame had mixed dtypes, produced one
Python object per element, and failed outright on `datetime64` columns because
`tolist()` yields `Timestamp` objects no transformer handled.

The codec dispatches on dtype instead and tags each column with a `kind`, so
decoders never need the document version to interpret a value.

## Payload encodings

| Type | Encoding | Notes |
|---|---|---|
| DataFrame, Series | parquet (zstd) | Needs pyarrow; falls back to JSON |
| ndarray, non-object dtype | `.npy` | `allow_pickle=False`, always |
| Everything else | JSON | Uses the normal transformer |

Parquet cannot represent duplicate column names or columns of arbitrary Python
objects. Rather than fail the write, `_try_dataframe_to_parquet` returns `None`
and the caller falls back to a JSON payload — larger, but it round-trips
anything the transformer understands.

`allow_pickle=False` on `.npy` is a security property, not a performance one: a
payload must never be able to execute code when it is read back.

## Coming soon: lazy value materialisation

`read_archive(nodes=[...])` currently takes an explicit list of nodes to
materialise. The natural next step is for values to be fetched on **first
access** instead, so a caller can open a multi-gigabyte archive, touch two
nodes, and pay for only those — without knowing in advance which two.

This is not yet implemented, and the reason is worth recording.

There are around a dozen sites that read `NodeAttributes.VALUE`, spread across
`computeengine.py`, `visualization.py` and `ui/viewmodel.py`. Two of them are
bulk reads:

- `Computation.to_dict` — `nx.get_node_attributes(self.dag, VALUE)`
- the DataFrame view used by `Computation.to_df`

A lazy sentinel stored in the `VALUE` attribute would be materialised by those
two the moment anyone called them, quietly defeating the entire feature while
appearing to work. Any implementation therefore needs, in order:

1. A single accessor all value reads route through.
2. Bulk readers taught to either skip or explicitly force unmaterialised nodes.
3. A decision on what `state()` reports for a node whose value has not yet been
   fetched — it is not `UNINITIALIZED`, but claiming `UPTODATE` for something
   not in memory is its own kind of lie.
4. A policy for what happens when the underlying archive is closed or moved
   while a lazy computation is still alive.

Step 3 is the genuinely open design question. Until it is settled, the explicit
`nodes=` form gives most of the benefit with none of the ambiguity, which is
why it shipped first.

## Known gaps

- The docs under `docs/user/` contain `pycon` blocks that are not executed by
  CI. They have drifted before — this work corrected two `States` enum values
  that had been wrong in `serializing_computations.md`. Wiring the markdown
  into pytest via `--doctest-glob` would prevent recurrence, but a number of
  existing examples would need fixing first.
- `mkdocs.yml` references `development/UI_EXTRA_PLAN.md`,
  `UI_EXTRA_AS_BUILT.md` and `UI_EXTRA_UX_REVIEW.md`, none of which exist, while
  `AIRFLOW_EXTRA_PLAN.md` exists but is not in the nav.
