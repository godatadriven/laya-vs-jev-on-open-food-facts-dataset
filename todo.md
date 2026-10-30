# Todo, 30 October 2026

The plan for the day.
Scope, questions and expected output are in [README.md](README.md).

## 1. Predictions (11:15, 10 min)

- [x] Fill in the predictions table in the README: majority baseline, Laya accuracy and Jev accuracy for each question.

Done when every cell has a number.
Guess the baselines too, before looking at the class distribution.

## 2. Data (11:25, 45 min)

- [ ] Look at the distinct values of `food_groups_tags` first and pick the level-1 groups. Open Food Facts mixes level-1 and level-2 tags in one list.
- [ ] Query `hf://datasets/openfoodfacts/product-database/food.parquet` with DuckDB, reading only the columns that are needed. If reading over HTTP is slow, download the file (8 GB) and query it locally.
- [ ] Filter: `lang = 'en'`, an English `product_name` and `ingredients_text`, `nova_group` not null, `nutriscore_grade` in a to e, exactly one level-1 food group.
- [ ] Build one `state` per product: name, ingredients, and nutrition per 100 g (energy, fat, saturated fat, sugars, salt, fibre, proteins) from `nutriments`.
- [ ] Draw 100_000 rows with a fixed seed, mark rows 1-10_000 as `calibration` and 10_001-100_000 as `evaluation`, save as parquet.
- [ ] Print the class distribution for each question and write the majority baselines down.
- [ ] Read 20 states by eye.

Done when the sample is on disk and the 20 states look like something a person could answer from.
If a level-1 group has fewer than 30 rows in the evaluation split, say so in the results rather than resampling to hide it.

## 3. Environment (12:10, 20 min)

- [ ] Start Laya on 0.4.18 instead of the 0.4.15 from the XKE demo, the batch endpoint is newer. The first start after the version bump runs without `HF_HUB_OFFLINE=1` in case the weights changed:

  ```sh
  LAYA_HOST=127.0.0.1 LAYA_PORT=8765 LAYA_MODELS=english LAYA_JEV_STRICT=1 USE_TF=0 \
  uv run --no-project --with "laya[serve]==0.4.18" -- laya-serve
  ```

- [ ] Check that Laya runs on MPS and not on CPU, and send one warm-up request: the first MPS call pays about 13 s of kernel compilation.
- [ ] Send `request.json` to both systems and compare the shape of the two responses.
- [ ] Find out which concrete version `jev-latest` resolves to, pin it, and write it down.
- [ ] Read TypeSafe's terms on publishing benchmarks. The answer decides whether Jev numbers can leave Xebia.

Done when both systems answer the demo request and both versions are recorded.

## 4. Harness (12:30, 1 h 30 min)

- [ ] One client for the `/v1/systemone` protocol, with `base_url`, `model` and the API key as the only things that change between systems.
- [ ] Laya through `/v1/systemone/batch`, up to 64 states per call.
- [ ] Jev with a small pool of concurrent requests, well under the published limit of 1200 requests per minute.
- [ ] The three questions defined once and shared by both systems, with the same option order.
- [ ] Output, one row per product, system and question: `id`, `split`, `system`, `model_version`, `question`, `answer`, `probs` (every option), `latency_ms`, input tokens when the response reports them.
- [ ] Keep the raw JSON responses as well, so the analysis never needs a second call.
- [ ] Dry run on 20 rows for each system before the full run.

Done when the 20-row dry run produces the same columns for both systems.

## 5. Runs (14:00, 1 h)

- [ ] Laya on all 100_000 rows. Record wall-clock time and rows per second.
- [ ] Jev on all 100_000 rows. Record latency p50 and p95, and input tokens per row.
- [ ] Count failed requests and rows with a truncated state, and keep them in the results instead of silently dropping them.

Done when both result files have 100_000 products times 3 questions.

## 6. Analysis (15:00, 1 h 30 min)

All reported numbers come from the evaluation split.

- [ ] Accuracy per question for both systems, next to the majority baseline.
- [ ] Confusion matrix for `food_group`, for both systems.
- [ ] Agreement between Laya and Jev per question, and 10 disagreements read by eye.
- [ ] Laya calibration: fit the temperature on the calibration split, then compare the reliability before and after on the evaluation split. Laya ships `laya/calibrate.py` for this. Jev's reliability as it comes.
- [ ] Laya accuracy against coverage: sort by confidence and plot accuracy on the top share of rows.
- [ ] Cost: € per million rows for Jev from the measured input tokens.

Done when the results table in the README can be filled in.

## 7. Write-up (16:30, 1 h)

- [ ] Fill in the results table in the README.
- [ ] Add the accuracy-against-coverage chart.
- [ ] Write three failure modes, each with real rows from the sample.
- [ ] Put the predictions next to the results and say where they were wrong.

Done when someone who was not here today can read the README and know what was found.

## Stretch, only if step 6 is finished by 16:00

- [ ] Shuffle the order of the `food_group` options with `option_order` and rerun Laya on the evaluation split. Report how many answers change.
