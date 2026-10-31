"""Fine-tune Laya on the training data exported from the laptop, on the DGX Spark's GPU.

    uv run python train.py check                                     # the GPU, the base checkpoint and the data
    uv run python train.py train --run smoke --limit 256 --epochs 1  # every step on 256 products, and a time estimate
    uv run python train.py train --run v1                            # the full run

A run writes `out/<run>/`, the checkpoint `laya.load` reads, with `run.json` and `train.log` beside it, and packs it
into `out/<run>.tar` with a `.sha256`, for the trip back to the laptop.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
LAYA_VERSION = "0.4.18"  # the release the laptop serves the checkpoint with
BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"  # the English checkpoint the laptop's Laya serves
WEIGHTS = "model.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def base_checkpoint() -> Path:
    """The pinned base checkpoint, downloaded once into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    files = ["rl_agent_config.json", WEIGHTS, "tokenizer/*", "encoder/*"]  # what `laya.train` loads
    return Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=files))


def training_data() -> dict:
    """The manifest of the exported data, after checking the data is the file it describes."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    if sha256(DATA / "train.jsonl") != manifest["train_sha256"]:
        raise SystemExit("data/train.jsonl does not match data/manifest.json: copy both from the laptop again")
    return manifest


def check_laya() -> None:
    installed = importlib.metadata.version("laya")
    if installed != LAYA_VERSION:
        raise SystemExit(f"laya {installed} is installed, but the laptop serves {LAYA_VERSION}")


def check(args: argparse.Namespace) -> None:
    import torch

    print(f"python {platform.python_version()} on {platform.machine()}, torch {torch.__version__}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"torch sees no CUDA device (built for CUDA {torch.version.cuda})")
        free, total = torch.cuda.mem_get_info()
        major, minor = torch.cuda.get_device_capability()
        print(f"{torch.cuda.get_device_name()}, compute capability {major}.{minor}, CUDA {torch.version.cuda}")
        print(
            f"{free / 1e9:.0f} of {total / 1e9:.0f} GB free; kernels built for {' '.join(torch.cuda.get_arch_list())}"
        )

    # One real kernel on the device, in the half precision training runs in, against the CPU
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    half = torch.float16 if args.device == "cuda" else torch.float32
    error = ((a.to(args.device, half) @ b.to(args.device, half)).float().cpu() - a @ b).abs().max().item()
    print(f"matmul on {args.device} matches the CPU to {error:.3f}")
    if error > 1.0:
        raise SystemExit("the device computes wrong results")

    check_laya()
    print(f"laya {LAYA_VERSION}")
    base = base_checkpoint()
    print(f"base checkpoint {BASE_REPO}@{BASE_REVISION[:7]} in {base}")
    manifest = training_data()
    print(f"data: {manifest['products']:,} {manifest['split']} products, exported {manifest['created']}")

    # The base checkpoint answers the first product on the device, the path inference takes
    import laya

    with (DATA / "train.jsonl").open() as file:
        row = json.loads(file.readline())
    agent = laya.load(str(base), device=args.device)
    for question, answer in agent.system_one(row["state"], row["questions"])["answers"].items():
        # Keyed the way the training data labels it: a choice by its label, a noul by bool, a score by level
        if answer["type"] == "noul":
            probs = {True: answer["noul"], False: 1 - answer["noul"]}
        else:
            probs = {int(k) if answer["type"] == "score" else k: p for k, p in answer["probabilities"].items()}
        best = max(probs, key=probs.__getitem__)
        print(f"product {row['id']}, {question}: {best} ({probs[best]:.2f}), labelled {row['expected'][question]}")
    print("ready to train")


class Tee:
    """Writes to the terminal and to the run's train.log at once, so the log comes back with the checkpoint."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def train(args: argparse.Namespace) -> None:
    import torch
    from laya import train as laya_train

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run):
        raise SystemExit("--run takes lowercase letters, digits, - and _")
    out = OUT / args.run
    if out.exists():
        raise SystemExit(f"out/{args.run} already exists; pick another --run name")
    check_laya()
    manifest = training_data()
    base = base_checkpoint()

    config = laya_train.TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,  # 32 x 2 by default: the effective batch of 64 Laya publishes
        loss="soft-ce",  # Laya measured no gain from the RL term on typed decisions (#741 in its repo)
        shuffle_options=("choice",),  # food_group options in a new random order every epoch: no position prior
        calib_max=1500,  # held out from training to fit the temperatures: about 500 per question instead of 133
        gradient_checkpointing=False,  # memory is plenty on the Spark, so do not trade speed for it
        seed=0,
    )

    out.mkdir(parents=True)
    log = (out / "train.log").open("w")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)

    data = DATA / "train.jsonl"
    if args.limit:
        rows = data.read_text().splitlines(keepends=True)[: args.limit]
        data = OUT / f".{args.run}.jsonl"  # beside the checkpoint folder, so it is not packed with it
        data.write_text("".join(rows))
    products = args.limit or manifest["products"]

    # The training loop alone is timed too, so a short run can tell how long a long one takes
    loop = {}
    train_model = laya_train.train_model

    def timed_train_model(*positional, **keywords):
        start = time.perf_counter()
        try:
            return train_model(*positional, **keywords)
        finally:
            loop["seconds"] = time.perf_counter() - start

    laya_train.train_model = timed_train_model  # `finetune` looks it up in the module when it calls it

    started, start = now(), time.perf_counter()
    print(
        f"{args.run}: {products:,} products, {args.epochs} epoch(s), base {BASE_REPO}@{BASE_REVISION[:7]}", flush=True
    )
    try:
        summary = laya_train.finetune(
            data=str(data), model_dir=str(base), output_dir=str(out), config=config, device=args.device
        )
    finally:
        if args.limit:
            data.unlink()
    seconds = time.perf_counter() - start

    # Laya holds out min(calib_max, calib_frac of the items) for the temperatures and trains on the rest
    per_item = loop["seconds"] / (summary["train_items"] * config.epochs)
    all_items = (summary["train_items"] + summary["calibration_items"]) / products * manifest["products"]
    full_train_items = all_items - min(config.calib_max, int(config.calib_frac * all_items))
    run = {
        "run": args.run,
        "started": started,
        "finished": now(),
        "seconds": round(seconds, 1),
        "training_loop_seconds": round(loop["seconds"], 1),
        "train_items_per_second": round(1 / per_item, 1),
        "products": products,
        "train_items": summary["train_items"],
        "calibration_items": summary["calibration_items"],
        "skipped": summary["skipped"],
        "epoch_loss": summary["epoch_loss"],
        "temperature": summary["temperature"],
        "temperature_by_options": summary["temperature_by_options"],
        "base": {"repo": BASE_REPO, "revision": BASE_REVISION},
        "data": manifest | {"limit": args.limit},
        "config": asdict(config),
        "device": args.device,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "laya": importlib.metadata.version("laya"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if args.device == "cuda" else None,
        },
        "weights_sha256": sha256(out / WEIGHTS),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(
        f"{args.run}: {duration(seconds)} in all, the training loop {duration(loop['seconds'])}, "
        f"{run['train_items_per_second']} items/s"
    )
    if args.limit:
        print(
            f"a run on all {manifest['products']:,} products: about {duration(per_item * full_train_items)} per epoch"
        )
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    log.close()  # complete before it is packed
    archive = pack(out)
    print(f"packed {archive.relative_to(HERE)}, {archive.stat().st_size / 1e9:.1f} GB; copy it with its .sha256")


def pack(out: Path) -> Path:
    """`out/<run>.tar`: the final checkpoint without `checkpoint_latest`, a copy of the same weights."""
    archive = out.parent / f"{out.name}.tar"

    def final_only(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if "checkpoint_latest" in Path(info.name).parts else info

    with tarfile.open(archive, "w") as tar:
        tar.add(out, arcname=out.name, filter=final_only)
    (out.parent / f"{archive.name}.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )  # `sha256sum -c` reads it
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help in [("check", "check the GPU, the base checkpoint and the data"), ("train", "fine-tune and pack")]:
        command = commands.add_parser(name, help=help)
        command.add_argument("--device", default="cuda", help="cuda on the Spark; mps or cpu to try it elsewhere")
        if name == "train":
            command.add_argument("--run", required=True, help="the folder under out/, for example v1")
            command.add_argument("--epochs", type=int, default=4, help="default 4, as Laya's published recipe")
            command.add_argument("--micro-batch", type=int, default=32, help="items per forward pass (default 32)")
            command.add_argument("--grad-accum", type=int, default=2, help="micro batches per update (default 2)")
            command.add_argument("--limit", type=int, help="only the first N products, for a short trial run")
    args = parser.parse_args()
    if args.command == "check":
        check(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
