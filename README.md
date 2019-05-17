# Public repo for DeepDrugCoder (DDC).

## Installation
- Clean conda cache and unused packages to avoid conflicts `conda clean --all`.
- Create a Python3.6 environment and install `rdkit` by running `conda create -c rdkit -n ddc_env rdkit`.
- Install CUDA dependencies by `conda install cudatoolkit=9.0 && conda install cudnn`.
- Run `python setup.py install` to install alongside with pip dependencies.
- Add the environment in the drop-down list of jupyter by `python -m ipykernel install --user --name ddc_env --display-name "ddc_env (python_3.6.7)"`.

## Usage (within your ddc environment)
- `from ddc_pub import ddc_v3 as ddc`
