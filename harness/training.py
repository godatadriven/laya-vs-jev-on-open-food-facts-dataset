"""Fine-tuning Laya: the training data for `finetune/train.py`, and the checkpoint it packs.

`export` writes the calibration split as the JSONL rows `laya.train` reads, into `finetune/data/`.
`install` checks a packed checkpoint and unpacks it to `data/checkpoints/laya-ft/`, where `just laya-ft`
serves it from.
"""

import datetime as dt
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import duckdb

from harness.questions import GRADES, QUESTIONS, questions_sha256, truth_of

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample.parquet"
EXPORT = ROOT / "finetune" / "data"
CHECKPOINT = ROOT / "data" / "checkpoints" / "laya-ft"  # the one fine-tuned checkpoint the experiment serves
WEIGHTS = "model.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def expected(food_group: str, ultra_processed: bool, nutriscore: str) -> dict:
    """The answers a sample row should get, in the form `laya.train` reads: a score is a level index."""
    truth = truth_of(food_group, ultra_processed, nutriscore)
    return {
        "food_group": truth["food_group"],
        "ultra_processed": truth["ultra_processed"] == "yes",
        "nutriscore": GRADES.index(truth["nutriscore"]),
    }


def export(split: str = "calibration") -> Path:
    """Write the products of `split` to `finetune/data/train.jsonl`, with a manifest `train.py` checks it against."""
    rows = (
        duckdb.connect()
        .execute(
            "SELECT id, state, food_group, ultra_processed, nutriscore FROM read_parquet(?) WHERE split = ? ORDER BY id",
            [str(SAMPLE), split],
        )
        .fetchall()
    )
    EXPORT.mkdir(parents=True, exist_ok=True)
    path = EXPORT / "train.jsonl"
    with path.open("w") as file:
        for id, state, *labels in rows:
            row = {"id": id, "state": state, "questions": QUESTIONS, "expected": expected(*labels)}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sample_sha256": sha256(SAMPLE),
        "split": split,
        "products": len(rows),
        "ids": [rows[0][0], rows[-1][0]],
        "questions_sha256": questions_sha256(QUESTIONS),
        "train_sha256": sha256(path),
    }
    (EXPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def install(archive: Path) -> Path:
    """Check a checkpoint archive against its `.sha256` and the questions, then unpack it."""
    archive = archive.resolve()
    recorded = archive.with_name(archive.name + ".sha256").read_text().split()[0]
    if sha256(archive) != recorded:
        raise RuntimeError(f"{archive.name} does not match its .sha256: the download is incomplete or changed")
    if CHECKPOINT.exists():
        raise RuntimeError(f"{CHECKPOINT.relative_to(ROOT)} already holds a checkpoint; move it away first")

    staging = CHECKPOINT.with_name(CHECKPOINT.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    (unpacked,) = staging.iterdir()  # the archive holds one folder, named after the training run
    run = json.loads((unpacked / "run.json").read_text())

    # The checkpoint learned these questions, wording and option order included: it must be asked the same ones
    if json.dumps(json.loads((unpacked / "questions.json").read_text())) != json.dumps(QUESTIONS):
        raise RuntimeError("the checkpoint was trained on other questions than request.json")
    if run["data"]["questions_sha256"] != questions_sha256(QUESTIONS):
        raise RuntimeError("the checkpoint was trained on data exported with other questions")
    if sha256(unpacked / WEIGHTS) != run["weights_sha256"]:
        raise RuntimeError(f"{WEIGHTS} does not match the digest its training run recorded")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    unpacked.rename(CHECKPOINT)
    staging.rmdir()
    return CHECKPOINT
