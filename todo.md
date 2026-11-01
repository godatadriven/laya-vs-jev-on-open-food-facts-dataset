# Todo, 30 October 2026

The plan for the day.
Scope, questions and expected output are in [README.md](README.md).

## 1. Predictions (11:15, 10 min)

- [x] Fill in the predictions table in the README: majority baseline, Laya accuracy and Jev accuracy for each question.

Done when every cell has a number.
Guess the baselines too, before looking at the class distribution.

## 2. Data (11:25, 45 min)

- [x] Look at the distinct values of `food_groups_tags` first and pick the level-1 groups. Open Food Facts mixes level-1 and level-2 tags in one list.
- [x] Query `hf://datasets/openfoodfacts/product-database/food.parquet` with DuckDB, reading only the columns that are needed. If reading over HTTP is slow, download the file (8 GB) and query it locally.
- [x] Filter: `lang = 'en'`, an English `product_name` and `ingredients_text`, `nova_group` not null, `nutriscore_grade` in a to e, exactly one level-1 food group.
- [x] Build one `state` per product: name, ingredients, and nutrition per 100 g (energy, fat, saturated fat, sugars, salt, fibre, proteins) from `nutriments`.
- [x] Draw 100_000 rows with a fixed seed, mark rows 1-10_000 as `calibration` and 10_001-100_000 as `evaluation`, save as parquet.
- [x] Print the class distribution for each question and write the majority baselines down.

Done when the sample is on disk and the 20 states look like something a person could answer from.
If a level-1 group has fewer than 30 rows in the evaluation split, say so in the results rather than resampling to hide it.

## 3. Environment (12:10, 20 min)

- [x] Start Laya on 0.4.18 instead of the 0.4.15 from the XKE demo, the batch endpoint is newer. The first start after the version bump runs without `HF_HUB_OFFLINE=1` in case the weights changed:

  ```sh
  LAYA_HOST=127.0.0.1 LAYA_PORT=8765 LAYA_MODELS=english LAYA_JEV_STRICT=1 USE_TF=0 \
  uv run --no-project --with "laya[serve]==0.4.18" -- laya-serve
  ```

- [x] Check that Laya runs on MPS and not on CPU, and send one warm-up request: the first MPS call pays about 13 s of kernel compilation.
- [x] Send `request.json` to both systems and compare the shape of the two responses.
- [x] Find out which concrete version `jev-latest` resolves to, pin it, and write it down.
- [ ] Read TypeSafe's terms on publishing benchmarks. The answer decides whether Jev numbers can leave Xebia.

Done when both systems answer the demo request and both versions are recorded.

## 4. Harness (12:30, 1 h 30 min)

- [x] One client for the `/v1/systemone` protocol, with `base_url`, `model` and the API key as the only things that change between systems.
- [x] Laya through `/v1/systemone/batch`, up to 64 states per call.
- [x] Jev with a small pool of concurrent requests, well under the published limit of 1200 requests per minute.
- [x] The three questions defined once and shared by both systems, with the same option order.
- [x] Output, one row per product, system and question: `id`, `split`, `system`, `model_version`, `question`, `answer`, `probs` (every option), `latency_ms`, input tokens when the response reports them.
- [x] Keep the raw JSON responses as well, so the analysis never needs a second call.
- [x] Dry run on 20 rows for each system before the full run.

Done when the 20-row dry run produces the same columns for both systems.

## 5. Runs (14:00, 1 h)

- [x] Laya on all 100_000 rows, on the DGX Spark (`just run laya` there, as [finetune/README.md](finetune/README.md) describes). Record wall-clock time and rows per second; the laptop's rows per second come from the first look.
  66 min on the Spark, 25.2 products per second, on 1 November; the laptop answers 5.0.
- [x] Jev on all 100_000 rows. Record latency p50 and p95, and input tokens per row.
  On the evaluation split: p50 162 ms, p95 238 ms, 888 input tokens per product on average (median 867).
  2.9 hours of running, spread over 10.4 hours because the laptop slept; the harness's clock stops during sleep, so no latency includes it.
- [x] Count failed requests and rows with a truncated state, and keep them in the results instead of silently dropping them.
  No product failed for any system; 23 Jev products needed a second attempt.
  Laya and `laya-ft` truncate 4_038 of the 270_000 evaluation rows: 2.8% of `food_group`, 0.9% of `nutriscore`, 0.8% of `ultra_processed`.

Done when both result files have 100_000 products times 3 questions.

## 6. Analysis (15:00, 1 h 30 min)

All reported numbers come from the evaluation split.

- [x] Accuracy per question for both systems, next to the majority baseline.
- [x] Confusion matrix for `food_group`, for both systems.
- [x] Agreement between Laya and Jev per question, and 10 disagreements read by eye.
- [x] Laya calibration: fit the temperature on the calibration split, then compare the reliability before and after on the evaluation split. Laya ships `laya/calibrate.py` for this. Jev's reliability as it comes.
- [x] Laya accuracy against coverage: sort by confidence and plot accuracy on the top share of rows.
- [x] Cost: € per million rows for Jev from the measured input tokens.

All of it is in [results.ipynb](notebooks/results.ipynb), for all three systems.

Done when the results table in the README can be filled in.

## 7. Write-up (16:30, 1 h)

- [x] Fill in the results table in the README.
- [x] Add the accuracy-against-coverage chart.
- [ ] Write three failure modes, each with real rows from the sample.
- [x] Put the predictions next to the results and say where they were wrong.

Done when someone who was not here today can read the README and know what was found.

## Fine-tuned Laya, added 31 October

How it works is in [finetune/README.md](finetune/README.md).

- [ ] Fill in the `laya-ft` column of the predictions table in the README, before the training run starts.
- [x] Export the calibration split as training data: `just ft-data`.
- [x] Copy the training code and data to the DGX Spark, and check its GPU there.
- [x] Smoke run on 256 products: 18.1 items per second, about 26 min per epoch.
- [x] Full run, `just finetune train --run v1`, in `tmux`: 87 min in all, 21 min per epoch, 22.7 items per second; accuracy on the 1_500 held-out items from 0.444 to 0.878.
- [ ] Read `v1/run.json`: the peak memory, the loss of each epoch, the fitted temperatures and the digest of the weights.
- [x] Copy the repo to the Spark with `just spark-push spark almaz/laya-vs-jev`, then `uv sync --locked` and `just ft-install v1` there.
  `v1` was trained in the older folder, so copy its archive into the repo first: `mkdir -p ~/almaz/laya-vs-jev/finetune/out && cp ~/almaz/laya-finetuning/out/v1.tar* ~/almaz/laya-vs-jev/finetune/out/`.
- [x] On the Spark, `just laya-ft`, then `just run laya-ft full --split evaluation`: `laya-ft@c4b23c4`, 59 min, 25.1 products per second.
- [x] On the Spark, `just laya`, then `just run laya`.
- [x] On the laptop, `just spark-pull spark almaz/laya-vs-jev`, `just ft-install v1` and `just build`.
- [x] Compare the Spark's Laya answers with the laptop's on the 1_024 products of the first look.
  Same batches, so only the device differs: 18 `food_group`, 8 `ultra_processed` and 2 `nutriscore` answers change, and no probability moves by more than 0.09.
  The laptop is right on 6 of the changed `food_group` answers and the Spark on 2; the same for `ultra_processed`.
- [x] Accuracy of `laya-ft` per question next to Laya and Jev, for products whose name is in the calibration split and for the rest.

Done when `results.parquet` has `laya-ft` answers for the 90_000 evaluation products.

## Stretch, only if step 6 is finished by 16:00

- [ ] Shuffle the order of the `food_group` options with `option_order` and rerun Laya on the evaluation split. Report how many answers change.
