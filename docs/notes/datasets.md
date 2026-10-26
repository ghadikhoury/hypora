# Real dataset snapshots (S1.6)

Spec: §20 (real data sources at runtime), §4.3. Decisions: D-015 (real OpenML/UCI data only),
D-021 (only datasets with a clear license). Code: `scripts/snapshot_dataset.py` (fetching),
`hypora/storage/snapshots.py` (manifest, license rule, file integrity).

```bash
make snapshot SOURCE=openml ID=46918     # = uv run --locked python scripts/snapshot_dataset.py openml 46918
make snapshot SOURCE=uci ID=17
uv run --locked python scripts/snapshot_dataset.py list
uv run --locked python scripts/snapshot_dataset.py verify
```

## What a snapshot is

`data/snapshots/<source>-<id>-<name>.parquet` (e.g. `openml-46918-credit-g.parquet`) plus one entry
in `data/snapshots/manifest.json`. `data/` is git-ignored: snapshots are re-fetched per machine, and
the manifest records enough to check that a re-fetch is the same data.

| Manifest field | Meaning |
|---|---|
| `source`, `dataset_id`, `title`, `version` | OpenML ID and dataset version; UCI ID and its "last updated" date (UCI has no versions) |
| `source_url`, `download_url` | the dataset's page; the exact file downloaded |
| `license`, `license_declared`, `license_url`, `license_evidence` | the recognized license ID, the declaration as the source states it, and where it was read |
| `citation`, `original_data_url` | as the source gives them (UCI: built from the dataset DOI) |
| `target` | the source's default target column, when it names exactly one |
| `dropped_columns` | columns the source marks as not features, each with the reason |
| `sha256`, `bytes` | the stored Parquet file |
| `source_sha256` | the bytes downloaded from the source |
| `retrieved_at`, `tool_versions` | when, and with which Python/pandas/PyArrow |

The stored file's SHA-256 is what the controller records as a run's dataset fingerprint (the
profile's `fingerprint`), so `hypora status` can name a run's real source by looking the
fingerprint up in the manifest. Re-fetching with the same library versions gives byte-identical
files (checked: a forced re-fetch reproduced the recorded SHA-256), and `source_sha256` detects a
change at the source independently of the Parquet writer.

## The license rule (D-021)

A dataset is stored only if the license its source declares is one we can name.

- **OpenML:** the dataset's `licence` field must be a recognized license. Most older uploads,
  including classic datasets such as iris (61), breast-w (15), adult (1590), and credit-g (31),
  declare `Public`, which names no license, and are refused. The refusal points to the UCI
  original when OpenML lists one. Curated newer uploads usually name their license, e.g. the
  TabArena datasets 46904-46958 (credit-g as 46918, bank-marketing as 46910; mostly CC BY 4.0).
- **UCI:** the dataset page's License section must link a Creative Commons license. The UCI API
  has no license field; every page sampled states CC BY 4.0.
- Recognized: CC0, the six CC 4.0/3.0/2.5/2.0/1.0 variants (with a version: the terms differ
  between versions), "Public Domain" (an explicit dedication), PDDL, ODbL, ODC-By, DbCL, MIT,
  Apache-2.0. Refused: `Public`, `See source`, versionless CC names, anything unrecognized.
  `recognize_license` and `license_from_url` hold the table; tests pin each case. License links
  require an HTTP(S) URL on `creativecommons.org` (or `www.creativecommons.org`) with the license
  in its path; lookalike hosts and license strings in queries or fragments do not establish terms.
- Non-commercial licenses (CC BY-NC*) are clear licenses and are accepted; their ID says NC.
- Only `active` OpenML datasets are stored (`in_preparation` and `deactivated` are refused).

A refusal writes nothing. If a step needs a real dataset that cannot be obtained this way, the
agent stops and asks Ghadi with its IDs (PLAN.md §0.3 trigger 9); it never substitutes synthetic
data.

## Fetching

- HTTPS only, to `www.openml.org`, `data.openml.org`, and `archive.ics.uci.edu`, checked before
  the request, before dispatching every redirect hop, and on the final response; metadata capped
  at 8 MiB, data at 512 MiB. Checking only the final response would already have contacted a
  forbidden intermediate host, even when the chain returns to an approved host.
- **OpenML:** `GET /api/v1/json/data/<id>` (description: status, licence, target, row-ID and
  ignored attributes, Parquet URL), checked before any data is downloaded; then the Parquet file.
  When the response's ETag is a plain MD5 (single-part S3 object), the download is verified
  against it. Row-ID and ignored attributes are dropped (OpenML's own default view).
- **UCI:** `GET /api/dataset?id=<id>` (metadata, variables with roles, `data_url`), the dataset
  page (license), then `data.csv`. Identifier variables are dropped. A declared missing-value
  marker outside pandas' defaults (e.g. `?`) is read as missing, and the entry says so.
- An already-stored dataset is verified against its recorded SHA-256 and not downloaded again;
  a changed file is an error until `--force` re-fetches it, which also reports a changed source.

### Why the REST APIs and not the `openml` package

`openml` was pre-approved for this step. It was tried (0.15.1, the newest release, January 2025)
and works with our pinned pandas 3.1 / NumPy 2.5, but it was not used:

- its `get_data()` caches each dataset as a pickle and loads it back on later calls (spec §14.3:
  no unpickling of artifacts in the platform), so it could only be used for metadata and download;
- for those, the Parquet URL is a private attribute (`_parquet_url`), and the client mutates a
  global cache configuration;
- it adds six transitive packages (minio, pycryptodome, argon2-cffi, liac-arff, xmltodict, tqdm)
  for what the documented JSON API gives with two GETs.

The REST path downloads byte-identical Parquet (compared against the client's download of 46918).

## Checked by hand (2026-10-26)

`openml 46918` (credit-g, CC BY 4.0, 1000 x 21, categorical dtypes kept), `openml 46954` (QSAR fish
toxicity, regression), `uci 17` (breast cancer diagnostic; `ID` dropped, target `Diagnosis`),
`uci 53` (iris) stored; `openml 61` refused (`Public`, with the UCI hint); `openml 46926` refused
(`in_preparation`); `uci 999999` not found; a tampered file caught by `verify` and by a re-fetch;
`--force` reproduced the recorded SHA-256. Tests use a fake HTTP layer
(`tests/unit/test_snapshot_script.py`), never the network.
