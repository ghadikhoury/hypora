# Canonical JSON and hashing (S0.2)

Every hash Hypora records goes through one utility, `hypora/core/hashing.py`:

| Function | Input | Used for |
|---|---|---|
| `canonical_hash(value)` | JSON-compatible value or pydantic model | experiment contracts, approved plans, approval artifacts, any `Record.content_hash()` |
| `sha256_hex(data)` | raw bytes | artifact files, dataset fingerprints, checkpoints |

**Canonical form** (what `canonical_json` emits, then SHA-256, lowercase hex):

1. Object keys sorted by code point; separators `,` and `:` with no whitespace.
2. UTF-8, no ASCII escaping (`"éthanol"`, not `"éthanol"`).
3. Floats normalized: NaN and ±Infinity are **rejected**; `-0.0` → `0`; an integral float with
   |x| ≤ 2^53 is written as an integer (so `1.0` and `1` hash the same); every other float uses
   Python's shortest round-trip repr (`0.1`, `1e-07`, `2.5e+20`).
4. Only JSON types are accepted. Sets, bytes, non-string keys, and arbitrary objects raise
   `TypeError` instead of being coerced.
5. Pydantic models are first dumped with `model_dump(mode="json")`: enums become their values,
   datetimes ISO 8601 strings (schema datetimes are normalized to UTC first, so equal instants hash
   equally), aliases are used (`"from"`, not `"from_"`).

It is close to RFC 8785 (JCS); the differences (key order by code point instead of UTF-16 code
units, Python float repr instead of ECMAScript's) only matter for astral-plane keys and exponent
formatting, which are irrelevant here and stay deterministic.

**Hashes are never stored inside the record they describe.** `ExperimentContract.hash` is a property
that recomputes `content_hash()`. A stored hash therefore can't go stale after an edit or be forged in
the JSON; the DB keeps it in its own column (`experiment_contracts.hash`, `approvals.artifact_hash`),
and anything that loads a record can recompute and compare.

**Pinned values.** `tests/unit/schemas/test_hashing.py` pins a hash computed independently with
`printf '%s' '{"a":true,"b":[1.5,"é"],"c":2}' | sha256sum`, and the hash of the §7.2 data-contract
fixture. If a dependency upgrade changes canonicalization, those tests fail instead of every stored
approval silently becoming invalid.
