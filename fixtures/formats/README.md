# Format fixtures

## `mini.safetensors`

A deterministic Safetensors file used to test the reader through both unit and
integration tests.

Contents:

| Tensor | Dtype | Shape | Values | Offsets |
|---|---|---|---|---|
| `alpha` | F32 | `[2, 2]` | 1.0, 2.0, 3.0, 4.0 | `[0, 16]` |
| `beta` | BF16 | `[3]` | 0.5, -1.0, 2.0 | `[16, 22]` |
| `empty` | F32 | `[0]` | none | `[0, 0]` |

Metadata:

```json
{
  "generator": "seldon-vault-lite M1 fixture",
  "version": "1"
}
```

File size: 281 bytes. Header size: 251 bytes.

SHA-256:
`caca3b030faeb4ded52275d54c2fca0f2ee0d6b1aeeb8ab1819e0785b1f89ca7`

## Generation

The fixture was generated outside the product. The recipe is:

```python
import json, struct, hashlib, pathlib

header = {
    "__metadata__": {
        "generator": "seldon-vault-lite M1 fixture",
        "version": "1",
    },
    "alpha": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    "beta": {"dtype": "BF16", "shape": [3], "data_offsets": [16, 22]},
    "empty": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
}
header_bytes = json.dumps(
    header, sort_keys=True, separators=(",", ":")
).encode("ascii")
data = (
    struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    + struct.pack("<3H", 0x3f00, 0xbf80, 0x4000)
)
payload = struct.pack("<Q", len(header_bytes)) + header_bytes + data
pathlib.Path("fixtures/formats/mini.safetensors").write_bytes(payload)
print(hashlib.sha256(payload).hexdigest())
```

Python is a laboratory tool only. It is not part of the product and is not
invoked by the binary or the test suite.

## Purpose and limits

The fixture proves:

- header parsing and deterministic key ordering;
- F32 little-endian decoding;
- BF16 widening;
- zero-sized tensor handling;
- fingerprinting;
- truncation and corruption rejection.

It does not prove model parity, tokenization, or inference correctness.
