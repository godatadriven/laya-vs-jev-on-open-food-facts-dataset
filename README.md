# Innovation Day, 30 October 2026: Laya vs Jev on Open Food Facts

One day, two decision models, one public dataset with its own labels.
The working plan for the day is in [todo.md](todo.md).

## The idea

In September 2026 a new class of models appeared: System One models, also called decision models.
They *do not* generate text.
They read the input once and answer typed questions (`choice`, `noul` for yes/no, `score` for ordered levels) with a probability per option, in a single forward pass.

For a data engineering practice this is not a chatbot feature.
It is a new kind of transformation: a cheap, typed function with a confidence attached, that can run on every row of a table, like a UDF in Spark or SQL.


| | Jev | Laya |
| --- | --- | --- |
| Vendor | TypeSafe AI | Convai Innovations |
| Access | Closed, hosted API, US | Apache 2.0, self-hosted |
| Model | Architecture not published | ModernBERT-large encoder, 421M (English checkpoint) |
| Price | $0.042 per million input tokens (vendor figure) | Own hardware |
| Known weak spots | Numbers and dates (vendor docs) | Base checkpoints near chance zero-shot on typed decisions, over-confident as shipped, 512-token context |

**The question of the day:** can Laya, running on our own machine, replace Jev on every row of a table, and how much accuracy does that cost?

Jev's published benchmarks score it against the averaged answers of two large models, not against ground truth.
This day measures both systems against labels that already exist in the data, which is a number nobody has published yet.

The day also produces the "own numbers" that the Innovation Projects pitch is missing: see `app/data/materials/ideas/sovereign-decision-models.md`, section "Свои числа для питча".

## Scope

### Data

Open Food Facts, the official parquet export on Hugging Face: `openfoodfacts/product-database`, file `food.parquet` (8 GB, 4.88M products).

The sample is 2000 English products with all three labels present, drawn with a fixed seed:

- rows 1-1000 are the calibration split, used to fit Laya's temperature and the cascade threshold;
- rows 1001-2000 are the evaluation split, where every reported number comes from.

### Three questions, one request per product

| Question | Type | Ground truth | Why it is in |
| --- | --- | --- | --- |
| `food_group` | `choice` | Level-1 tag in `food_groups_tags` (about 9 groups) | The main metric: a human-curated taxonomy, not a rule |
| `ultra_processed` | `noul` | `nova_group == 4` | NOVA in Open Food Facts is computed by rules from ingredients, so this is "model vs regex" |
| `nutriscore` | `score` | `nutriscore_grade`, a to e | Expected to fail: the grade is a function of numbers, a known weak spot. It shows where the boundary is |

### What is measured

- Accuracy per question and per system on the evaluation split, next to the majority-class baseline.
- Where Laya and Jev disagree, and examples of each kind of disagreement.
- Calibration: reliability of Laya's confidence before and after temperature fitting, and Jev's as shipped.
- Accuracy against coverage for Laya: how accurate it is on the share of rows it is most confident about.
- Throughput and cost: rows per second for Laya on an M3 Pro laptop, latency and € per million rows for Jev from measured token counts.

### The cascade

Laya answers every row, and rows below a confidence threshold go to Jev.
The headline number of the day: **what share of rows has to leave for the US API for the cascade to match Jev's accuracy.**
That is the argument "sovereign by default, API by exception", measured instead of claimed.
The cascade needs no extra calls: it is computed from the two runs.

### Out of scope

- Generative LLM baselines, Strands Decider, Clef and the OpenAI Decisions API.
- Fine-tuning Laya.
- The multilingual checkpoint and non-English products.
- Any cloud, Kubernetes or table-format work: that is the Innovation Project, not the day.
- Publishing Jev numbers outside Xebia until TypeSafe's terms on benchmarks are read.

## Predictions, written before the runs

Fill these in before the first request is sent.

| Question | Majority baseline | Laya accuracy | Jev accuracy |
| --- | --- | --- | --- |
| `food_group` | | | |
| `ultra_processed` | | | |
| `nutriscore` | | | |

Share of rows sent to Jev for the cascade to match Jev: ___

## Expected output

By the end of the day this folder holds:

1. **The results table**: accuracy per question for Laya, Jev, the cascade and the majority baseline, plus Laya rows per second and Jev € per million rows.
2. **One chart**: Laya accuracy against coverage, with the cascade threshold marked.
3. **The cascade number**: the share of rows sent to Jev at Jev's accuracy.
4. **Three failure modes**, each with real rows from the sample as examples.
5. **Predictions against results**: where the guesses above were wrong, and by how much.
6. **One line for the pitch**: own accuracy and speed, ready to paste into the Innovation Projects draft.

A negative result counts.
"Laya zero-shot reaches X% on food groups, fine-tuning is required" is a finding and a line for the pitch.

## Caveats known in advance

- Laya's base checkpoints are over-confident as shipped, so raw confidence is not a threshold until it is fitted.
- The English Laya checkpoint reads 512 tokens, so long ingredient lists are cut.
- Option order changes Laya's answers; the stretch item in [todo.md](todo.md) measures it.
- Laya and Jev figures above come from vendor pages and secondary reviews, not from our measurements.
- The sample is a derived database under the ODbL licence of Open Food Facts; keep the attribution if it is ever published.

## References

- [XKE S3 LLMs in 2026 (demo part)](https://xebiagroup.sharepoint.com/:v:/r/sites/Xebia-Data/_layouts/15/stream.aspx?id=%2Fsites%2FXebia%2DData%2FShared%20Documents%2FKnowledge%20sharing%2FXKE%20recordings%2FBuilding%20an%20LLM%20Series%2FS3%20%2D%20LLMs%20in%202026%20Transformers%2C%20Mixture%20of%20Experts%20and%20System%20One%20Solutions%2D20261027%5F180332%2DMeeting%20Recording%2Emp4&share=cQp2QNbgTjUHSL2BO4UthXgnEgUCdv9aSXdeELyGZ7w7%5FZE1Hg)
- [Laya on GitHub](https://github.com/NandhaKishorM/laya) - checkpoints, serving, calibration, limitations
- [Open Food Facts on Hugging Face](https://huggingface.co/datasets/openfoodfacts/product-database)
- [TypeSafe Jev pricing, limits and benchmarks](https://www.developersdigest.tech/blog/typesafe-jev-system-one-models-release-guide-2026)
- [Laya: the open-source Jev alternative, benchmarked](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
