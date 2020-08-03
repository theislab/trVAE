"""trVAE - Regularized Conditional Variational Autoencoders"""
import warnings
warnings.filterwarnings(action='ignore')

from . import models
from . import utils as tl
from . import data as dl
from . import plotting as pl
from . import metrics as mt

__author__ = ', '.join([
    'Mohsen Naghipourfar',
    'Mohammad Lotfollahi'
])

__email__ = ', '.join([
    'mohsen.naghipourfar@gmail.com',
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
])
