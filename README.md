# Public repo for DeepDrugCoder (DDC).

#[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)](https://www.python.org/downloads/release/python-360/) [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


<p align="center">
<a href="https://www.gnu.org/licenses/gpl-3.0"><img alt="Build Status" src="https://img.shields.io/badge/License-GPLv3-blue.svg"></a>
#[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
#[![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)](https://www.python.org/downloads/release/python-360/) 
#[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
</p>

<p align="center">
<a href="https://travis-ci.org/python/black"><img alt="Build Status" src="https://travis-ci.org/python/black.svg?branch=master"></a>
<a href="https://black.readthedocs.io/en/stable/?badge=stable"><img alt="Documentation Status" src="https://readthedocs.org/projects/black/badge/?version=stable"></a>
<a href="https://coveralls.io/github/python/black?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/python/black/badge.svg?branch=master"></a>
<a href="https://github.com/python/black/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://pypi.org/project/black/"><img alt="PyPI" src="https://black.readthedocs.io/en/stable/_static/pypi.svg"></a>
<a href="https://pepy.tech/project/black"><img alt="Downloads" src="https://pepy.tech/badge/black"></a>
<a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


### Installation
- Create a predefined Python3.6 conda environment by `conda create env file==ddc_env.yml`
- Run `python setup.py install` to install pip dependencies and add the package to the Python path.
- Add the environment in the drop-down list of jupyter by `python -m ipykernel install --user --name ddc_env --display-name "ddc_env (python_3.6.7)"`.

### Usage (within ddc_env)
- `from ddc_pub import ddc_v3 as ddc`

Detailed breakdown of the full model:
![alt text](img/detailed_model.png)
