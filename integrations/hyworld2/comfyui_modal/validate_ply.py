from __future__ import annotations

import hashlib
import json
import pathlib
import sys

TYPE_BYTES = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1}


def validate_gaussian_ply(path: pathlib.Path) -> dict:
    with path.open("rb") as stream:
        header_lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("truncated PLY header")
            header_lines.append(line)
            if line == b"end_header\n":
                break
        header_bytes = stream.tell()

    header = b"".join(header_lines).decode("ascii").splitlines()
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("expected binary_little_endian PLY 1.0")
    vertex_line = next(line for line in header if line.startswith("element vertex "))
    vertex_count = int(vertex_line.rsplit(" ", 1)[1])
    properties = [line.split() for line in header if line.startswith("property ")]
    stride = sum(TYPE_BYTES[parts[1]] for parts in properties)
    expected_bytes = header_bytes + vertex_count * stride
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"invalid payload length: expected {expected_bytes}, got {actual_bytes}")
    return {
        "path": str(path.resolve()),
        "vertices": vertex_count,
        "properties": len(properties),
        "bytes_per_vertex": stride,
        "bytes": actual_bytes,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(validate_gaussian_ply(pathlib.Path(sys.argv[1])), indent=2))
