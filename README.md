# Public repo for DeepDrugCoder (DDC).

## Installation
- Create a `python 3.6` environment and install `rdkit` by running `conda create -c rdkit -n ddc rdkit`.
- Install CUDA dependencies by `conda install cudatoolkit=9.0 && conda install cudnn`.
- Run `python setup.py install` to install alongside with pip dependencies.

## Usage (within your ddc environment)
- `from ddc_pub import ddc_v3 as ddc`
