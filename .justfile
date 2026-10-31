# A venv activated for another project must not leak into this one
unexport VIRTUAL_ENV

notebook := "data-sampling.ipynb"
kernel := "laya-vs-jev"

# List the recipes
default:
    @just --list

# Register this project's .venv as the Jupyter kernel the notebooks run on
kernel:
    uv run --locked python -m ipykernel install --sys-prefix --name {{kernel}} --display-name "Python (.venv, {{kernel}})"

# Open JupyterLab on the project venv
lab: kernel
    uv run --locked jupyter lab {{notebook}}

# Run the notebook top to bottom and save its outputs, so GitHub shows them
execute: kernel
    uv run --locked jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name={{kernel}} {{notebook}}
