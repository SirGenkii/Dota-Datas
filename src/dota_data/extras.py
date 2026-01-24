from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


EXTRAS_SCHEMA = pa.schema(
    [
        ("match_id", pa.int64()),
        ("radiant_gold_adv", pa.string()),
        ("radiant_xp_adv", pa.string()),
        ("picks_bans", pa.string()),
    ]
)


def _json_dumps_or_none(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return None


def _iter_wrapped_matches_from_array_json(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream items from a top-level JSON array file.
    Each item is expected to be a wrapped match: {"json": {...}, ...}.
    """
    import ijson  # local import to keep dependency optional at import time

    with path.open("rb") as f:
        for item in ijson.items(f, "item"):
            if isinstance(item, dict):
                yield item


def _iter_wrapped_matches_from_chunk_files(root: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate wrapped matches from chunk files (matches_chunk*.json) under a directory (recursive).
    """
    for fp in sorted([p for p in root.glob("**/matches_chunk*.json") if p.is_file()]):
        try:
            with fp.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            continue
        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict):
                    yield item


def _extras_row(wrapped: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    match = wrapped.get("json")
    if not isinstance(match, dict):
        return None
    mid = match.get("match_id")
    if not isinstance(mid, int):
        return None
    return {
        "match_id": int(mid),
        "radiant_gold_adv": _json_dumps_or_none(match.get("radiant_gold_adv")),
        "radiant_xp_adv": _json_dumps_or_none(match.get("radiant_xp_adv")),
        "picks_bans": _json_dumps_or_none(match.get("picks_bans")),
    }


def write_extras_parquet(
    *,
    raw: Path,
    out: Path,
    batch_size: int = 5000,
    overwrite: bool = False,
) -> Path:
    """
    Build `extras.parquet` from raw sources without loading everything in memory.

    `raw` can be:
    - a big combined JSON file (top-level array)
    - a directory containing chunk files (matches_chunk*.json)
    """
    raw = Path(raw)
    out = Path(out)
    if out.suffix == ".parquet":
        out_path = out
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "extras.parquet"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    if raw.is_dir():
        it = _iter_wrapped_matches_from_chunk_files(raw)
    else:
        it = _iter_wrapped_matches_from_array_json(raw)

    writer: Optional[pq.ParquetWriter] = None
    buffer: List[Dict[str, Any]] = []
    written = 0
    try:
        writer = pq.ParquetWriter(out_path, EXTRAS_SCHEMA, compression="zstd")
        for wrapped in it:
            row = _extras_row(wrapped)
            if row is None:
                continue
            buffer.append(row)
            if len(buffer) >= batch_size:
                table = pa.Table.from_pylist(buffer, schema=EXTRAS_SCHEMA)
                writer.write_table(table)
                written += len(buffer)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=EXTRAS_SCHEMA)
            writer.write_table(table)
            written += len(buffer)
    finally:
        if writer is not None:
            writer.close()

    print(f"[extras] wrote {written} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed extras.parquet (adv arrays + picks/bans) from raw sources.")
    parser.add_argument("--raw", default="data/raw/data_v2.json", help="Raw JSON file (array) OR a directory of chunk files.")
    parser.add_argument("--out", default="data/processed", help="Output dir (default) or output parquet file path.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Rows per parquet write batch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extras.parquet.")
    args = parser.parse_args()

    write_extras_parquet(raw=Path(args.raw), out=Path(args.out), batch_size=args.batch_size, overwrite=args.overwrite)
