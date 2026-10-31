"""Command line: `python -m harness run laya|laya-ft|jev [--run NAME] [--split SPLIT] [--limit N]`, `build`,
`export-training` and `install-checkpoint`."""

import argparse
import asyncio
from pathlib import Path

from harness import training
from harness.runs import ROOT, SYSTEMS, build, run


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    send = commands.add_parser("run", help="send the sample to one system; rerunning continues where it stopped")
    send.add_argument("system", choices=SYSTEMS)
    send.add_argument("--run", default="full", help="the folder under data/runs (default: full)")
    send.add_argument("--split", choices=["calibration", "evaluation"], help="one split only (default: both)")
    send.add_argument("--limit", type=int, help="only the first N products, in id order")

    results = commands.add_parser("build", help="turn a run's raw logs into results.parquet")
    results.add_argument("--run", default="full", help="the folder under data/runs (default: full)")

    commands.add_parser("export-training", help="write the calibration split to finetune/data/, the training data")
    install = commands.add_parser(
        "install-checkpoint", help="check and unpack a checkpoint archive from finetune/train.py"
    )
    install.add_argument("archive", type=Path, help="out/<run>.tar, with its .sha256 beside it")

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args.system, args.run, split=args.split, limit=args.limit))
    elif args.command == "build":
        print(build(args.run))
    elif args.command == "export-training":
        print(training.export().relative_to(ROOT))
    else:
        print(training.install(args.archive).relative_to(ROOT))


if __name__ == "__main__":
    main()
