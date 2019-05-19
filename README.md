# Public repo for DeepDrugCoder (DDC).

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)](https://www.python.org/downloads/release/python-360/) [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


### Installation
- Create a predefined Python3.6 conda environment by `conda create env file==ddc_env.yml`
- Run `python setup.py install` to install pip dependencies and add the package to the Python path.
- Add the environment in the drop-down list of jupyter by `python -m ipykernel install --user --name ddc_env --display-name "ddc_env (python_3.6.7)"`.

### Usage (within ddc_env)
- `from ddc_pub import ddc_v3 as ddc`

The complete model is a combination of 3 submodels:
<img src="https://bitbucket.astrazeneca.net/users/kjmv588/repos/ddc_pub/browse/img/model.png" width="48">

Detailed breakdown of the full model:
![alt text](img/detailed_model.png)
