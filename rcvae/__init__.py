"""RCVAE - Regularized Conditional Variational Autoencoders"""

from .models import *
from .data_loader import prepare_and_load_celeba, resize_image
from . import plotting


__author__ = ', '.join([
    'Mohammad Lotfollahi',
    'Mohsen Naghipourfar'
])

__email__ = ', '.join([
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
    'mohsen.naghipourfar@gmail.com'
])

from get_version import get_version
__version__ = get_version(__file__)
del get_version




