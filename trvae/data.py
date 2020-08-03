import os

from PIL import Image

import scanpy as sc


def read(file_path):
    return sc.read(file_path)
