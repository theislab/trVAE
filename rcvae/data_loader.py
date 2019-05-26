import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import anndata
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


def prepare_and_load_celeba(file_path, attr_path,
                            gender='Male', attribute='Smiling',
                            max_n_images=None,
                            restore=True,
                            save=True,
                            img_resize=64,
                            verbose=True):
    data_path = os.path.dirname(file_path)
    zip_filename = os.path.basename(file_path).split(".")[0]
    if restore and os.path.exists(os.path.join(data_path, f"celeba_{attribute}_{img_resize}.h5ad")):
        return sc.read(os.path.join(data_path, f"celeba_{attribute}_{img_resize}.h5ad"))

    def load_attr_list(file_path):
        indices = []
        attributes = []
        with open(file_path) as f:
            lines = f.read().splitlines()
            columns = lines[1].split(" ")
            columns.remove('')
            if max_n_images is not None:
                max_n = max_n_images
            else:
                max_n = len(lines)
            for i in range(2, max_n):
                elements = lines[i].split()
                indices.append(elements[0])
                attributes.append(list(map(int, elements[1:])))
        attr_df = pd.DataFrame(attributes)
        attr_df.index = indices
        attr_df.columns = columns
        if verbose:
            print(attr_df.shape[0])
        return attr_df

    images = []
    zfile = zipfile.ZipFile(file_path)
    counter = 0
    attr_df = load_attr_list(attr_path)
    print(len(attr_df.index.tolist()))
    indices = []
    for filename in attr_df.index.tolist():
        ifile = zfile.open(os.path.join(f"{zip_filename}/", filename))
        image = Image.open(ifile)
        image = image.resize((img_resize, img_resize), Image.NEAREST)
        image = np.reshape(image, (img_resize, img_resize, 3))
        if max_n_images is None:
            images.append(image)
            indices.append(filename)
            counter += 1
            if verbose and counter % 1000 == 0:
                print(counter)
        else:
            if counter < max_n_images:
                images.append(image)
                indices.append(filename)
                counter += 1
                if verbose and counter % 1000 == 0:
                    print(counter)
            else:
                break
    images = np.array(images)
    if verbose:
        print(images.shape)
    images_df = pd.DataFrame(images.reshape(-1, np.prod(images.shape[1:])))
    images_df.index = indices

    if save:
        data = anndata.AnnData(X=images_df.values)
        print(data.shape, attr_df.shape)
        data.obs['labels'] = attr_df[gender].values
        data.obs['condition'] = attr_df[attribute].values
        sc.write(filename=os.path.join(data_path, f"celeba_{attribute}_{img_resize}.h5ad"), adata=data)
    return data


def resize_image(images, img_size):
    images_list = []
    for i in range(images.shape[0]):
        image = cv2.resize(images[i], (img_size, img_size), cv2.INTER_NEAREST)
        images_list.append(image)
    return np.array(images_list)
