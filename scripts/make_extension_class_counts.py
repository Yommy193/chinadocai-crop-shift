"""Create class weights for the additive FCOS recognition manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from oracle_vit.utils import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    processed = (ROOT / args.processed_dir).resolve()
    output = (ROOT / args.output).resolve()
    vocab = json.loads((processed / "vocab.json").read_text(encoding="utf-8"))
    rows = []
    with (processed / "crops.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "train":
                rows.append(row)
    counts = Counter(int(row["class_id"]) for row in rows)
    reference_root = processed / "glyph_reference_full"
    reference_counts = []
    for class_id, label in enumerate(vocab["labels"]):
        reference_counts.append(len(list((reference_root / str(label)).glob("*.png"))))
    vector = [counts.get(class_id, 0) + reference_counts[class_id] for class_id in range(len(vocab["labels"]))]
    payload = {
        "schema_version": 1,
        "definition": "FCOS-extension training rows plus the unchanged full-vocabulary reference glyph pool",
        "counts": vector,
        "train_rows": len(rows),
        "reference_samples": sum(reference_counts),
        "vocab_sha256": file_sha256(processed / "vocab.json"),
        "source_manifest_sha256": file_sha256(processed / "crops.jsonl"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "train_rows": len(rows), "classes": len(vector), "reference_samples": sum(reference_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
