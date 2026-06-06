#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""
Load 4 hallucination-detection datasets and produce full_data_out.json
following the exp_sel_data_out schema.

Datasets:
  1. FActScore-ChatGPT  — biography + per-atom S/NS labels (PRIMARY)
  2. SummaCoz           — article + summary + consistency label (SECONDARY)
  3. FRANK              — article + summary + sentence faithfulness annotations
  4. XSumFaith          — summary + span-level hallucination type annotations
"""

import csv
import json
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUT_PATH = WORKSPACE / "full_data_out.json"


# ── 1. FActScore-ChatGPT ─────────────────────────────────────────────────────

def load_factscore() -> list[dict]:
    path = DATASETS_DIR / "factscore_data" / "factscore" / "data" / "labeled" / "ChatGPT.jsonl"
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            topic = d.get("topic") or d.get("input", "")
            bio_text = d.get("output", "")

            atoms: list[dict] = []
            for ann in (d.get("annotations") or []):
                for af in (ann.get("human-atomic-facts") or []):
                    label_raw = af.get("label", "")
                    if label_raw == "S":
                        label = 1
                    elif label_raw in ("NS", "Not Supported"):
                        label = 0
                    else:
                        continue  # skip Irrelevant
                    atoms.append({"atom_text": af["text"], "label": label})

            if len(atoms) < 5:
                continue  # require >= 5 labeled atoms

            # input: the biography text to assess for hallucinations
            # output: JSON string of gold atom labels
            examples.append({
                "input": bio_text,
                "output": json.dumps(atoms),
                "metadata_topic": topic,
                "metadata_model": "ChatGPT",
                "metadata_n_atoms": len(atoms),
                "metadata_n_supported": sum(1 for a in atoms if a["label"] == 1),
                "metadata_n_not_supported": sum(1 for a in atoms if a["label"] == 0),
                "metadata_task_type": "hallucination_detection",
            })

    logger.info(f"FActScore: {len(examples)} biographies with >=5 labeled atoms")
    return examples


# ── 2. SummaCoz ──────────────────────────────────────────────────────────────

def load_summacoz() -> list[dict]:
    # Combine train + validation splits
    train_path = DATASETS_DIR / "full_nkwbtb_SummaCoz_default_train.json"
    val_path = DATASETS_DIR / "full_nkwbtb_SummaCoz_default_validation.json"

    all_rows: list[dict] = []
    for p in (train_path, val_path):
        all_rows.extend(json.loads(p.read_text()))

    examples = []
    for i, r in enumerate(all_rows):
        label_raw = r.get("label")
        # SummaCoz encoding: 0 = inconsistent (hallucinated), 2 = consistent
        if label_raw == 0:
            label = "1"  # consistent (no inconsistency reason needed)
        elif label_raw == 2:
            label = "0"  # inconsistent (reason explains the hallucination)
        else:
            continue  # skip unexpected values

        article = r.get("article", "")
        summary = r.get("summary", "")
        reason = (r.get("reason") or "")[:400]

        # input: article + summary (the model must judge consistency)
        # output: consistency label (0=inconsistent, 1=consistent)
        examples.append({
            "input": f"Article:\n{article}\n\nSummary:\n{summary}",
            "output": label,
            "metadata_reason": reason,
            "metadata_origin_dataset": r.get("dataset", ""),
            "metadata_origin": r.get("origin", ""),
            "metadata_task_type": "consistency_detection",
            "metadata_row_index": i,
        })

    logger.info(f"SummaCoz: {len(examples)} article/summary pairs")
    return examples


# ── 3. FRANK ─────────────────────────────────────────────────────────────────

def load_frank() -> list[dict]:
    path = DATASETS_DIR / "frank_annotations.json"
    data = json.loads(path.read_text())

    examples = []
    for i, r in enumerate(data):
        article = r.get("article", "")
        summary = r.get("summary", "")
        sent_anns = r.get("summary_sentences_annotations") or []

        # Derive overall label: faithful if majority of annotators say no errors
        # sent_anns is a list of dicts: [{'annotator_0': [...], 'annotator_1': [...], ...}]
        def sentence_faithful(ann_dict: dict) -> bool:
            votes = [len(v) == 0 for v in ann_dict.values() if isinstance(v, list)]
            return sum(votes) > len(votes) / 2  # majority vote

        is_faithful = all(
            sentence_faithful(ann) for ann in sent_anns if isinstance(ann, dict)
        )
        label = "1" if is_faithful else "0"

        examples.append({
            "input": f"Article:\n{article}\n\nSummary:\n{summary}",
            "output": label,
            "metadata_model": r.get("model_name", ""),
            "metadata_split": r.get("split", ""),
            "metadata_sentence_annotations_json": json.dumps(sent_anns),
            "metadata_task_type": "faithfulness_detection",
            "metadata_row_index": i,
        })

    logger.info(f"FRANK: {len(examples)} article/summary pairs")
    return examples


# ── 4. XSumFaith ─────────────────────────────────────────────────────────────

def load_xsumfaith() -> list[dict]:
    path = DATASETS_DIR / "xsumfaith.csv"
    examples = []
    seen: set[tuple] = set()

    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            bbcid = row.get("bbcid", "")
            system = row.get("system", "")
            summary = row.get("summary", "")
            hall_type = row.get("hallucination_type", "")
            hall_span = row.get("hallucinated_span", "")

            # Deduplicate by (bbcid, system, summary[:80])
            key = (bbcid, system, summary[:80])
            if key in seen:
                continue
            seen.add(key)

            # label: NULL = faithful, otherwise hallucinated
            label = "0" if hall_type != "NULL" else "1"

            # input: the summary text (source article retrievable via bbcid)
            # output: hallucination label
            examples.append({
                "input": summary,
                "output": label,
                "metadata_bbcid": bbcid,
                "metadata_system": system,
                "metadata_hallucination_type": hall_type,
                "metadata_hallucinated_span": hall_span[:200],
                "metadata_task_type": "hallucination_detection",
                "metadata_row_index": i,
            })

    logger.info(f"XSumFaith: {len(examples)} deduplicated summary entries")
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    factscore = load_factscore()
    summacoz = load_summacoz()

    # FRANK discarded: all 2246 summaries are non-faithful (no balanced label distribution —
    # the dataset is an error-taxonomy resource, not a binary faithful/unfaithful benchmark).
    # XSumFaith discarded: no source article text (only bbcid reference), severely imbalanced
    # (87% hallucinated), span-level format not suited for lock-in coefficient experiments.

    data_out = {
        "datasets": [
            {"dataset": "FActScore-ChatGPT", "examples": factscore},
            {"dataset": "SummaCoz", "examples": summacoz},
        ]
    }

    total = sum(len(ds["examples"]) for ds in data_out["datasets"])
    logger.info(f"Total examples: {total}")

    OUT_PATH.write_text(json.dumps(data_out, indent=2, ensure_ascii=False))
    size_kb = OUT_PATH.stat().st_size / 1024
    logger.info(f"Wrote {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
