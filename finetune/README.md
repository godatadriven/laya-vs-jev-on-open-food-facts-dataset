# Fine-tuning Laya on the DGX Spark

This folder is copied to an NVIDIA DGX Spark as it is.
It fine-tunes the English Laya checkpoint on the calibration split and brings back one archive, which the laptop serves as `laya-ft`.

## What it trains

- **Base**: `convaiinnovations/laya` at revision `7b928d8`, the checkpoint the laptop's Laya serves, with `laya 0.4.18`.
- **Data**: the 10_000 products of the calibration split, three questions each, in the wording and option order of [request.json](../request.json).
  The evaluation split never leaves the laptop.
- **Recipe**: Laya's own `laya.train.finetune`, 4 epochs at an effective batch of 64 (32 per pass, 2 passes per update), as Laya publishes it.
- **Objective**: soft cross-entropy; Laya measured no gain from its RL term on typed decisions.
- **Option order**: the `food_group` options are shuffled every epoch, so the checkpoint cannot learn a position prior.
- **Calibration**: 1_500 of the 30_000 items are held out from training to fit the temperatures, about 500 per question.
- **Precision**: fp16 mixed precision on CUDA, which `laya.train` turns on by itself; gradient checkpointing is off, since the Spark has 128 GB.

## 1. On the laptop: export and copy

```sh
just ft-data          # finetune/data/train.jsonl and its manifest, from data/sample.parquet
just ft-push spark    # this folder without .venv/ and out/, to ~/laya-finetune on the host `spark`
```

`spark` is whatever name `ssh` knows the Spark by, for example from `~/.ssh/config`.
Any other way of copying works too: `data/manifest.json` holds the digest that `train.py` checks the data against.

## 2. On the Spark: check, try, train

The Spark needs to reach PyPI and Hugging Face.
Install `uv` once if it is not there: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```sh
cd ~/laya-finetune
uv sync --locked                                                 # Python 3.13, torch 2.14.1 for CUDA 13.0, laya 0.4.18
uv run python train.py check                                     # the GPU, the base checkpoint, the data, one answer
uv run python train.py train --run smoke --limit 256 --epochs 1  # every step on 256 products, and a time estimate
```

`check` downloads the base checkpoint (0.85 GB) and ends with `ready to train`.
The smoke run prints how long one epoch on all 10_000 products takes; on the M3 Pro it is about 2 h 15 min.

The full run goes in `tmux`, so it survives a dropped connection:

```sh
tmux new -s laya
uv run python train.py train --run v1
```

`Ctrl-b d` leaves it running, and `tmux attach -t laya` comes back to it.

Every run writes these, and refuses to start if `out/<run>/` already exists:

| Path | Holds |
| --- | --- |
| `out/<run>/` | the checkpoint `laya.load` reads, with the fitted temperatures in `rl_agent_config.json` and the questions in `questions.json` |
| `out/<run>/train_report.json` | Laya's before-and-after on the held-out items, and how the temperatures were fitted |
| `out/<run>/run.json` | timings, items per second, versions, GPU and peak memory, the settings, the data manifest and the digest of the weights |
| `out/<run>/train.log` | everything the run printed |
| `out/<run>.tar`, `out/<run>.tar.sha256` | all of the above but `checkpoint_latest/`, about 0.85 GB, for the trip back |

The before-and-after table at the end of the log is measured on the held-out training items.
It shows whether training worked; the results come from the evaluation split, on the laptop.

## 3. Back on the laptop: install and run

```sh
just ft-pull spark v1                     # out/v1.tar and its .sha256, into finetune/out/
just ft-install v1                        # checked and unpacked to data/checkpoints/laya-ft
just laya-ft                              # serves it as laya-ft on 127.0.0.1:8767
just run laya-ft full --split evaluation  # the 90_000 evaluation products, about 5 hours
```

`ft-install` refuses an archive that does not match its `.sha256`, a checkpoint trained on other questions than `request.json`, and a second checkpoint over the first.
A local checkpoint has no commit, so the harness pins its runs to the digest of the weights: `laya 0.4.18, laya-ft@<first 7 characters>`.

## When something on the Spark does not work

- **No access to Hugging Face**: copy the base checkpoint from the laptop's cache, `rsync -a ~/.cache/huggingface/hub/models--convaiinnovations--laya spark:.cache/huggingface/hub/`.
- **`uv sync` or the GPU check fails**: run the same steps in NVIDIA's PyTorch container, `nvcr.io/nvidia/pytorch:<yy.mm>-py3` from NGC, with `pip install laya==0.4.18` instead of `uv sync`.
  The container's library versions then differ from the lock; `run.json` records the ones that trained.
- **Out of memory**: `--micro-batch 16 --grad-accum 4` keeps the effective batch at 64 with half the memory.
