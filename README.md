# trVAE [![PyPI version](https://badge.fury.io/py/trvae.svg)](https://badge.fury.io/py/trvae) [![Build Status](https://travis-ci.org/theislab/trVAE.svg?branch=master)](https://travis-ci.org/theislab/trVAE)

<img align="center" src="./sketch/sketch.png?raw=true">

## Introduction
A Keras (with tensorflow backend) implementation of trVAE. trVAE is a deep generative model which learns mapping between multiple different styles (conditions). trVAE can be used for style transfer in images, single-cell perturbations response across celltypes, times and etc.  
<div float="left">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="80" >
  <img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" height="80">
</div>
<div float="right">
</div>

## Getting Started

## Installation

### Installation with pip
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
pip install flit
git clone https://github.com/theislab/trVAE
cd trVAE
flit install
```

## Examples

## Reproducing paper results:
In order to reproduce paper results visit [here](https://github.com/Naghipourfar/trVAE_reproducibility).


## References
Lotfollahi, Mohammad and Wolf, F. Alexander and Theis, Fabian J.
**"scGen predicts single-cell perturbation responses."**
Nature Methods, 2019. [pdf](https://rdcu.be/bMlbD)
