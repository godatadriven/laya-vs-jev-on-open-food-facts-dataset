# A venv activated for another project must not leak into this one
unexport VIRTUAL_ENV

kernel := "laya-vs-jev"
# The English checkpoint every run uses, whatever the Hugging Face Hub's latest is; /health reports it
laya_revision := "7b928d828b7b0e022f929d9bd2e44165aa270148"

# List the recipes
default:
    @just --list

# Prefix with HF_HUB_OFFLINE=1 once the weights are cached
# Start Laya on 127.0.0.1:8765; `just laya 8766 0` starts a second, non-strict one that reports truncation
laya port="8765" strict="1":
    LAYA_HOST=127.0.0.1 LAYA_PORT={{port}} LAYA_MODELS=english LAYA_REVISION={{laya_revision}} LAYA_JEV_STRICT={{strict}} USE_TF=0 \
    uv run --locked laya-serve

# Start a third Laya on 127.0.0.1:8767 that serves the fine-tuned checkpoint installed by `just ft-install`, as laya-ft
laya-ft port="8767":
    LAYA_HOST=127.0.0.1 LAYA_PORT={{port}} LAYA_MODELS=laya-ft LAYA_JEV_STRICT=1 USE_TF=0 HF_HUB_OFFLINE=1 \
    LAYA_EXTRA_MODELS='{"laya-ft": "{{justfile_directory()}}/data/checkpoints/laya-ft"}' \
    uv run --locked laya-serve

# Send the sample to laya, laya-ft or jev; rerunning continues where it stopped; Laya needs `just laya` running, laya-ft `just laya-ft`
run system run="full" *args:
    uv run --locked python -m harness run {{system}} --run {{run}} {{args}}

# Lint and format-check the harness and the fine-tuning script
lint:
    uvx ruff check harness finetune
    uvx ruff format --check harness finetune

# Turn the raw logs of a run into data/runs/<run>/results.parquet
build run="full":
    uv run --locked python -m harness build --run {{run}}

# Register this project's .venv as the Jupyter kernel the notebooks run on
kernel:
    uv run --locked python -m ipykernel install --sys-prefix --name {{kernel}} --display-name "Python (.venv, {{kernel}})"

# Open JupyterLab on the project venv
lab: kernel
    uv run --locked jupyter lab

# Run notebooks top to bottom and save their outputs, so GitHub shows them; systems-check and harness-check need `just laya`
execute +notebooks="notebooks/*.ipynb": kernel
    uv run --locked jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name={{kernel}} {{notebooks}}

# Write the calibration split to finetune/data/, the training data for `just finetune`
ft-data:
    uv run --locked python -m harness export-training

# Fine-tune Laya: `just finetune check`, then `just finetune train --run v1`; see finetune/README.md
finetune *args:
    uv run --locked python finetune/train.py {{args}}

# Check a packed training run and unpack it to data/checkpoints/laya-ft, where `just laya-ft` serves it
ft-install run:
    uv run --locked python -m harness install-checkpoint finetune/out/{{run}}.tar

# Copy the repo with the sample to the DGX Spark, without the Jev key: `just spark-push spark almaz/laya-vs-jev`
spark-push host dir:
    rsync -av --exclude /.git --exclude /.venv --exclude /.env --exclude /data/food.parquet --exclude '/data/sampling.duckdb*' \
        --exclude /data/runs --exclude /data/checkpoints --exclude /finetune/out \
        --exclude /.ruff_cache --exclude /.vscode --exclude .ipynb_checkpoints --exclude .DS_Store --exclude __pycache__ \
        ./ {{host}}:{{dir}}/

# Copy a run's raw logs and the packed checkpoint back from the Spark; a broken transfer continues where it stopped
spark-pull host dir run="full" checkpoint="v1":
    mkdir -p data/runs/{{run}} finetune/out
    rsync -av --partial {{host}}:{{dir}}/data/runs/{{run}}/ data/runs/{{run}}/
    rsync -av --partial "{{host}}:{{dir}}/finetune/out/{{checkpoint}}.tar*" finetune/out/
