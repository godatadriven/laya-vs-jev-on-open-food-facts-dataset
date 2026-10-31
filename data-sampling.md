# Data sampling

How to turn the local `food.parquet` into `data/sample.parquet`: the 100_000 English products described in the [README](README.md), split into calibration and evaluation rows.
This is step 2 of [todo.md](todo.md).
The same steps with charts, for the demo, are in [data-sampling.ipynb](notebooks/data-sampling.ipynb).

Everything runs in one DuckDB session. The expected numbers below come from the export downloaded on 30 October 2026 (4_884_051 products).

## What comes out

One parquet file, one row per product:

| Column | Content |
| --- | --- |
| `id` | 1 to 100_000, the order in which the seeded draw picked the product |
| `split` | `calibration` for ids 1-10_000, `evaluation` for ids 10_001-100_000 |
| `code` | Open Food Facts barcode |
| `name`, `ingredients` | English text with HTML removed |
| `energy_kcal`, `fat`, `saturated_fat`, `sugars`, `salt`, `fiber`, `proteins` | Per 100 g, null when the product does not report it |
| `food_group` | Level-1 tag, ground truth for `food_group` |
| `ultra_processed` | `nova_group = 4`, ground truth for `ultra_processed` |
| `nutriscore` | `A` to `E`, ground truth for `nutriscore` |
| `nova_group` | Raw NOVA group 1 to 4, kept for the analysis |
| `state` | The text both systems read |

## Steps

### 1. Set up

```sh
mkdir -p data
duckdb data/sampling.duckdb
```

The sample can always be rebuilt from these steps, so `data/` stays out of git, like `food.parquet`.
Run every SQL block below in that one session, in order.
Step 2 sets a session variable that steps 3 and 4 use; if you reopen DuckDB, run the `SET VARIABLE` again first.

### 2. Pick the level-1 food groups

`food_groups_tags` mixes levels in one list, for example `[en:fish-meat-eggs, en:meat, en:poultry]`.
In this export the level-1 tag is always the first element, so the level-1 groups are the distinct first elements.

```sql
SELECT food_groups_tags[1] AS level1, count(*) AS products
FROM 'data/food.parquet'
WHERE len(food_groups_tags) > 0
GROUP BY ALL
ORDER BY products DESC;
```

Expected: 11 groups, from `en:sugary-snacks` (307_969 products) down to `en:baby-foods-and-milks` (5_152).

```sql
SET VARIABLE level1 = (
  SELECT list(DISTINCT food_groups_tags[1])
  FROM 'data/food.parquet'
  WHERE len(food_groups_tags) > 0
);

SELECT len(list_filter(food_groups_tags, t -> list_contains(getvariable('level1'), t))) AS level1_tags,
       count(*) AS products
FROM 'data/food.parquet'
WHERE len(food_groups_tags) > 0
GROUP BY ALL;
```

Expected: a single row, `level1_tags = 1`, for all 1_527_487 products that have a food group.
If a row with 2 or more shows up, the first-element rule no longer holds and the level-1 list has to be written by hand.

### 3. Filter the candidates

The filter from the README: English, with a name and an ingredient list, all three labels present, exactly one level-1 group.

```sql
CREATE OR REPLACE MACRO clean(s) AS trim(
  replace(replace(replace(
    regexp_replace(regexp_replace(s, '<[a-zA-Z/][^>]*>', '', 'g'), '&(amp;)+', '&', 'g'),
    '&quot;', '"'), '&lt;', '<'), '&gt;', '>')
);

CREATE OR REPLACE TABLE candidates AS
WITH en AS (
  SELECT
    code,
    clean(list_filter(product_name, x -> x.lang = 'en')[1].text) AS name,
    clean(list_filter(ingredients_text, x -> x.lang = 'en')[1].text) AS ingredients,
    food_groups_tags,
    nova_group,
    nutriscore_grade,
    nutriments
  FROM 'data/food.parquet'
  WHERE lang = 'en'
    AND nova_group IS NOT NULL
    AND nutriscore_grade IN ('a', 'b', 'c', 'd', 'e')
)
SELECT *
FROM en
WHERE coalesce(name, '') <> ''
  AND coalesce(ingredients, '') <> ''
  AND len(list_filter(food_groups_tags, t -> list_contains(getvariable('level1'), t))) = 1
QUALIFY count(*) OVER (PARTITION BY code) = 1;

SELECT count(*) AS candidates, count(DISTINCT code) AS codes FROM candidates;
```

Expected: 274_159 candidates, all with a distinct code.

- `lang` is the product's main language.
  `product_name` and `ingredients_text` are lists of `{lang, text}`: take the `en` entry, not `main`, which can be in any language.
- `nutriscore_grade` also holds `unknown`, `not-applicable` and empty strings, hence the explicit list.
- A quarter of the ingredient lists wrap allergens in `<span class="allergen">`.
  `clean` removes the tags and decodes the few HTML entities, because markup is not part of the product and costs Laya tokens out of its 512.
  Allergens marked with underscores (`_milk_`) are an Open Food Facts convention and stay as they are.
- 61 barcodes appear twice in the full export.
  None of them survives the filter today; the `QUALIFY` drops both copies if that changes, rather than picking one at random.

### 4. Build the states

```sql
CREATE OR REPLACE MACRO num(x) AS rtrim(rtrim(printf('%.2f', x), '0'), '.');

CREATE OR REPLACE TABLE states AS
WITH labelled AS (
  SELECT
    code,
    name,
    ingredients,
    list_filter(nutriments, n -> n.name = 'energy-kcal')[1]."100g" AS energy_kcal,
    list_filter(nutriments, n -> n.name = 'fat')[1]."100g" AS fat,
    list_filter(nutriments, n -> n.name = 'saturated-fat')[1]."100g" AS saturated_fat,
    list_filter(nutriments, n -> n.name = 'sugars')[1]."100g" AS sugars,
    list_filter(nutriments, n -> n.name = 'salt')[1]."100g" AS salt,
    list_filter(nutriments, n -> n.name = 'fiber')[1]."100g" AS fiber,
    list_filter(nutriments, n -> n.name = 'proteins')[1]."100g" AS proteins,
    list_filter(food_groups_tags, t -> list_contains(getvariable('level1'), t))[1] AS food_group,
    nova_group = 4 AS ultra_processed,
    upper(nutriscore_grade) AS nutriscore,
    nova_group
  FROM candidates
)
SELECT
  *,
  'Product: ' || name || chr(10) ||
  'Ingredients: ' || ingredients || chr(10) ||
  'Nutrition per 100 g: ' || coalesce(nullif(concat_ws(', ',
    'energy ' || num(energy_kcal) || ' kcal',
    'fat ' || num(fat) || ' g',
    'saturated fat ' || num(saturated_fat) || ' g',
    'sugars ' || num(sugars) || ' g',
    'salt ' || num(salt) || ' g',
    'fibre ' || num(fiber) || ' g',
    'proteins ' || num(proteins) || ' g'
  ), ''), 'not reported') AS state
FROM labelled;
```

A state looks like this:

```text
Product: Candy Twists
Ingredients: Corn syrup, sugar, enriched wheat flour (wheat, niacin, iron, thiamin, riboflavin, folic acid), high fructose corn syrup, modified corn starch, water, glycerin, palm oil, salt, citric acid, artificial flavor, soy mono and diglycerides, potassium sorbate (preservative), artificial colors (red 40, blue 1).
Nutrition per 100 g: energy 349 kcal, fat 1.16 g, saturated fat 0 g, sugars 41.86 g, salt 0.7 g, fibre 0 g, proteins 2.33 g
```

- Read nutrients from the `"100g"` field, which is always in grams (kcal for energy).
  The `unit` field is the unit as typed on the label (`mg` for salt in almost half the candidates) and does not apply to `"100g"`.
- The nutrient names are `energy-kcal`, `fat`, `saturated-fat`, `sugars`, `salt`, `fiber` and `proteins`.
- Pick nutrients by name and never put the whole `nutriments` list in the state: it also holds `nova-group` and `nutrition-score-fr`, two of the labels.
  For the same reason `categories_tags`, `food_groups_tags`, `nova_groups_tags` and `nutrient_levels_tags` stay out of the state.
- A missing nutrient is left out of the state, not filtered out of the sample.
  Filtering on it would remove 18% of the rows, mostly products that do not report fibre.
- Numbers are rounded to 2 decimals with trailing zeros dropped.

### 5. Draw the sample and split it

```sql
CREATE OR REPLACE TABLE sample AS
SELECT
  row_number() OVER (ORDER BY md5('laya-vs-jev-2026-10-30' || code)) AS id,
  CASE WHEN id <= 10000 THEN 'calibration' ELSE 'evaluation' END AS split,
  *
FROM states
QUALIFY id <= 100000
ORDER BY id;
```

The seed is the string `laya-vs-jev-2026-10-30`.
Every candidate gets `md5(seed || code)`, and the 100_000 smallest hashes are the sample, in hash order.
Hash order is random, so the first 10_000 rows are already a random calibration split and no second shuffle is needed.

This is used instead of `USING SAMPLE ... REPEATABLE (seed)` because DuckDB's seeded sampling depends on the thread count and the scan order.
An md5 order picks the same rows on any machine and any DuckDB version, and can be repeated from Python if needed.

### 6. Check the sample

```sql
SELECT split, count(*) AS rows, min(id), max(id) FROM sample GROUP BY ALL ORDER BY 3;

SELECT count(*) AS rows, count(DISTINCT code) AS codes, count(*) FILTER (state IS NULL) AS null_states
FROM sample;

SELECT
  count(*) FILTER (energy_kcal IS NULL OR fat IS NULL OR saturated_fat IS NULL OR sugars IS NULL
                   OR salt IS NULL OR fiber IS NULL OR proteins IS NULL) AS missing_any_nutrient,
  count(*) FILTER (state LIKE '%Nutrition per 100 g: not reported') AS no_nutrition,
  count(*) FILTER (list_max([fat, saturated_fat, sugars, salt, fiber, proteins]) > 100
                   OR energy_kcal > 900
                   OR list_min([energy_kcal, fat, saturated_fat, sugars, salt, fiber, proteins]) < 0) AS implausible_nutrients,
  quantile_disc(length(state), [0.5, 0.9, 0.99]) AS state_chars_p50_p90_p99,
  count(*) FILTER (length(state) > 1500) AS longer_than_1500_chars
FROM sample;
```

Expected:

| Check | Value |
| --- | --- |
| `calibration` | 10_000 rows, ids 1-10_000 |
| `evaluation` | 90_000 rows, ids 10_001-100_000 |
| Distinct codes | 100_000, no null states |
| Missing at least one nutrient | 18_106 |
| No nutrition at all | 930 |
| Implausible nutrients | 328 |
| State length p50, p90, p99 | 366, 756, 1_425 characters |
| Longer than 1_500 characters | 797 |

- The 930 products without nutrition still have a Nutri-Score in the data.
  They stay in, with `not reported` in the state, and are worth a separate line in the `nutriscore` analysis.
- The 328 implausible rows (more than 100 g per 100 g, more than 900 kcal, or negative) stay in as well.
  `list_max` and `list_min` skip missing nutrients; `greatest` and `least` on DuckDB 1.1 return null as soon as one is missing, and undercount.
  That is what the data says, and they may show up as a `nutriscore` failure mode.
- Laya reads 512 tokens.
  At 3 to 4 characters per token that runs out somewhere around 1_500 to 2_000 characters, so expect about 1% of states to be cut.
  The harness counts the actual truncations from Laya's responses.

### 7. Class distribution and majority baselines

Evaluation split only, because every reported number comes from there.

```sql
SELECT food_group, count(*) AS rows, round(100 * rows / sum(rows) OVER (), 1) AS pct
FROM sample WHERE split = 'evaluation' GROUP BY ALL ORDER BY rows DESC;

SELECT ultra_processed, count(*) AS rows, round(100 * rows / sum(rows) OVER (), 1) AS pct
FROM sample WHERE split = 'evaluation' GROUP BY ALL ORDER BY rows DESC;

SELECT nutriscore, count(*) AS rows, round(100 * rows / sum(rows) OVER (), 1) AS pct
FROM sample WHERE split = 'evaluation' GROUP BY ALL ORDER BY rows DESC;
```

Expected majority baselines:

| Question | Majority class | Baseline |
| --- | --- | --- |
| `food_group` | `en:sugary-snacks` | 23.3% |
| `ultra_processed` | `true` | 67.0% |
| `nutriscore` | `E` | 30.8% |

- 10 food groups remain, not 9: `en:alcoholic-beverages` has 3 evaluation rows, because alcohol rarely gets a Nutri-Score.
  That is below 30, so it is reported as too few rows, not removed or resampled.
- `en:baby-foods-and-milks` does not survive the filter at all, so it is not one of the `food_group` options.
- Write the three baselines into the README results.

### 8. Read 20 states by eye

Read calibration rows, so the evaluation split stays unseen until the runs.

```sql
.mode line
SELECT id, food_group, ultra_processed, nutriscore, state
FROM sample WHERE split = 'calibration' ORDER BY id LIMIT 20;
.mode duckbox
```

For each state, ask whether a person could answer the three questions from it.
Look out for junk names (`40 SUME SEL POTATOES, WE SEE POTENTIAL 100% GREAT` is a real one), leftover markup, ingredient lists in another language despite the `en` tag, and nutrition lines that look off.
Write down anything that repeats, since it is a candidate failure mode for the write-up.

### 9. Save

```sql
COPY sample TO 'data/sample.parquet' (FORMAT parquet);

SELECT split, count(*) AS rows FROM 'data/sample.parquet' GROUP BY ALL ORDER BY split;
```

Expected: 10_000 `calibration` and 90_000 `evaluation` rows.

The sample only repeats on the same export, and the Hugging Face file is refreshed regularly.
Record which file it came from:

## Licence

The sample is a derived database of Open Food Facts, under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).
If it is ever published, it keeps the attribution: "Data from Open Food Facts, openfoodfacts.org, available under the ODbL".
