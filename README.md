# trVAE [![PyPI version](https://badge.fury.io/py/trvae.svg)](https://badge.fury.io/py/trvae) [![Build Status](https://travis-ci.org/theislab/trVAE.svg?branch=master)](https://travis-ci.org/theislab/trVAE)

<img align="center" src="./sketch/sketch.png?raw=true">

## Introduction
A Keras (tensorflow < 2.0) implementation of trVAE (transfer Variational Autoencoder) .

trVAE can be used for style transfer in images, predicting perturbations responses and batch-removal for single-cell RNA-seq.

* For pytorch implementation check [Here](https://github.com/theislab/trvaep)
## Getting Started

## Installation
Before installing trVAE package, we suggest you to create a new Python 3.6 (or 3.7) 
virtual env (or conda env) with the following steps:  

### 1.  Installing virtualenv
```bash
pip install virtualenv
```

### 2. Create a virtual with Python 3.6
```bash
virtualenv trvae-env --python=python3.6 
```

### 3. trVAE package installation
To install the latest version from PyPI, simply use the following bash script:
```bash
pip install trvae
```
or install the development version via pip: 
```bash
pip install git+https://github.com/theislab/trvae.git
```

or you can first install flit and clone this repository:
```bash
git clone https://github.com/theislab/trVAE
cd trVAE
pip install -r requirements
python setup.py install 
```

## Examples

* For perturbation prediction and batch-removal check this [example](https://nbviewer.jupyter.org/github/theislab/trVAE/blob/master/examples/trVAE_Haber.ipynb) from Haber et al.

## Reproducing paper results:
In order to reproduce paper results visit [here](https://github.com/Naghipourfar/trVAE_reproducibility).



