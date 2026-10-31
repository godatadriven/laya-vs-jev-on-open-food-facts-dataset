# Fine-tuning and measuring Laya on the DGX Spark

The NVIDIA DGX Spark does all of Laya's work: it fine-tunes `laya-ft`, and it runs the full measurement of Laya and `laya-ft`.
The repo goes there once, and the raw logs and the checkpoint come back.
Jev runs from the laptop, so its key never leaves it.

## What it trains

- **Base**: `convaiinnovations/laya` at revision `7b928d8`, the checkpoint `just laya` serves, with `laya 0.4.18`.
- **Data**: the 10_000 products of the calibration split, three questions each, in the wording and option order of [request.json](../request.json).
  No evaluation product is ever trained on.
- **Recipe**: Laya's own `laya.train.finetune`, 4 epochs at an effective batch of 64 (8 per pass, 8 passes per update), as Laya publishes it.
- **Objective**: soft cross-entropy; Laya measured no gain from its RL term on typed decisions.
- **Option order**: the `food_group` options are shuffled every epoch, so the checkpoint cannot learn a position prior.
- **Calibration**: 1_500 of the 30_000 items are held out from training to fit the temperatures, about 500 per question.
- **Precision**: fp16 mixed precision on CUDA, which `laya.train` turns on by itself; gradient checkpointing is off, so a pass is faster but needs more memory.

## How the data is split

The sample is counted in products, the trainer in items: an item is one product with one question.

```
100_000 products in the sample
├── 90_000 evaluation products   answered, never trained on; every reported number comes from here
└── 10_000 calibration products  the training data
      × 3 questions (food_group, ultra_processed, nutriscore)
      = 30_000 items
        ├── 28_500 train items     what the model learns from, for 4 epochs
        └──  1_500 held-out items  fit the temperatures, and give the before-and-after table
```

- **Items, not products**: the harness asks all three questions in one request, so it counts products.
  The trainer builds one sequence per question, the product's text with that question and its options, so every product becomes three training examples.
- **1_500 held out**: Laya holds out 10% of the items, but caps it at 400 by default, about 133 per question.
  `train.py` raises the cap to 1_500, about 500 per question, for steadier temperatures at the cost of 5% of the training data.
- **28_500 trained on**: the other 30_000 - 1_500 items.

The first line the trainer prints says the same:

```
train items 28500, calibration items 1500, eval items 1500 (calibration_slice), skipped {}, device cuda
```

- `eval items 1500 (calibration_slice)`: no separate evaluation file is given, so Laya scores the base checkpoint before training and the fine-tuned one after it on the 1_500 held-out items.
  That is the before-and-after table at the end of the log.
- `skipped {}`: no item was dropped.
  Laya drops an item when its state is empty, its label is not one of the options, or two options become identical once their descriptions are cut short.
  A truncated state is not dropped: it is trained on cut at 512 tokens, the way Laya reads it at inference.
- `device cuda`: the training runs on the Spark's GPU.

The next line is the optimizer budget: 446 updates per epoch at an effective batch of 64, 1_784 over the 4 epochs.
A "step" in the log is one pass of 8 items, so an epoch has 3_563 of them.

The before-and-after table shows whether training worked; it is not a result, for two reasons:

- The temperatures are fitted on the same 1_500 items the table scores, which flatters the calibration numbers (ECE and Brier).
  Accuracy is still fair: a temperature never changes which answer comes out on top.
- The items are held out one by one, not product by product, with a fixed seed.
  A product can have its `food_group` in training and its `nutriscore` held out, so the model may have read its text already, which makes the "after" score somewhat optimistic.

The results come from the 90_000 evaluation products.

## 1. On the laptop: copy the repo

```sh
just spark-push spark almaz/laya-vs-jev
```

It copies the repo with `data/sample.parquet` and the training data in `finetune/data/`, about 60 MB.
It leaves out `.env` with the Jev key, `food.parquet`, the virtual environment and the laptop's runs.
`spark` is whatever name `ssh` knows the Spark by, for example from `~/.ssh/config`.

## 2. On the Spark: set up

Once, if they are not there yet (`just` from Ubuntu's packages is too old for this repo's justfile):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install rust-just
```

Then, in the repo:

```sh
cd ~/almaz/laya-vs-jev
uv sync --locked     # the laptop's versions: Python 3.13, torch 2.14.1 (for CUDA 13.0 here), laya 0.4.18
just finetune check  # the GPU, the base checkpoint, the data, one answer; ends with "ready to train"
```

## 3. On the Spark: fine-tune

Everything long goes in `tmux`, so it survives a dropped connection: `tmux new -s laya`, `Ctrl-b d` to leave it, `tmux attach -t laya` to come back.

```sh
just finetune train --run smoke --limit 256 --epochs 1  # every step on 256 products, and the time per epoch
just finetune train --run v1                            # the full run, about 1 h 45 min on the office Spark
just ft-install v1                                      # unpacks it to data/checkpoints/laya-ft, for `just laya-ft`
```

The smoke run predicts 26 min per epoch on the office Spark, against about 2 h 15 min on the M3 Pro.

Every run writes these, and refuses to start if `finetune/out/<run>/` already exists:

| Path | Holds |
| --- | --- |
| `finetune/out/<run>/` | the checkpoint `laya.load` reads, with the fitted temperatures in `rl_agent_config.json` and the questions in `questions.json` |
| `finetune/out/<run>/train_report.json` | Laya's before-and-after on the held-out items, and how the temperatures were fitted |
| `finetune/out/<run>/run.json` | timings, items per second, versions, GPU and peak memory, the settings, the data manifest and the digest of the weights |
| `finetune/out/<run>/train.log` | everything the run printed |
| `finetune/out/<run>.tar`, `.tar.sha256` | all of the above but `checkpoint_latest/`, about 0.85 GB |

`ft-install` refuses an archive that does not match its `.sha256`, a checkpoint trained on other questions than `request.json`, and a second checkpoint over the first.
A local checkpoint has no commit, so the harness pins its runs to the digest of the weights: `laya 0.4.18, laya-ft@<first 7 characters>`.

## 4. On the Spark: measure

The same commands as on the laptop, each in a `tmux` window of its own:

```sh
just laya                                 # the published checkpoint, pinned to 7b928d8, on 127.0.0.1:8765
just laya-ft                              # the fine-tuned one, on 127.0.0.1:8767
just run laya                             # all 100_000 products
just run laya-ft full --split evaluation  # the 90_000 evaluation products
```

- Both servers listen on the Spark's own localhost only.
- A stopped run continues where it stopped: run the same command again.
- The two runs can go side by side: they share the GPU, but the Spark's speed is not a reported number.
- The laptop's speed is measured on the laptop, in the first look: 5.0 products per second on 1_024 products.

## 5. Back on the laptop: the results

```sh
just spark-pull spark almaz/laya-vs-jev  # data/runs/full/ with the Laya logs, and finetune/out/v1.tar
just ft-install v1                       # `just build` reads the checkpoint's digest and tokenizer from it
just run jev                             # if it has not run yet: about 2.8 hours
just build                               # data/runs/full/results.parquet, all three systems
```

## When something does not work

- **No access to Hugging Face from the Spark**: copy the base checkpoint from the laptop's cache, `rsync -a ~/.cache/huggingface/hub/models--convaiinnovations--laya spark:.cache/huggingface/hub/`.
- **Out of memory**: on the office Spark, two vLLM containers hold about 92 GB of the 121.7 GB, which leaves about 25 GB.
  A micro batch of 32 took 25.7 GB there and pushed the machine into swap; the default of 8 fits.
  Check `free -h` and `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` for other jobs before starting.
