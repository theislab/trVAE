import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import anndata
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_file(filename, backup_url=None,
              **kwargs):  # TODO : what if several fileS provided as csv or h5 e.g. x, label1, label2

    """
        Loads file in any of pandas, numpy or AnnData's extension.

        # Parameters
            filename: basestring
                name of the file which is going to be loaded.
            backup_url: basestring
                backup url for downloading data if the file with the specified `filename`
                does not exists.
            kwargs: dict
                dictionary of additional arguments for loading data with each package.

        # Returns
            The annotated matrix of loaded data.

        # Example
        ```python
        import scgen
        train_data_filename = "./data/train.h5ad"
        train_data = scgen.load_file(train_data_filename)
        ```

    """
    numpy_ext = {'npy', 'npz'}
    pandas_ext = {'csv', 'h5'}
    adata_ext = {"h5ad"}

    if not os.path.exists(filename) and backup_url is None:
        raise FileNotFoundError('Did not find file {}.'.format(filename))

    elif not os.path.exists(filename):
        d = os.path.dirname(filename)
        if not os.path.exists(d): os.makedirs(d)
        urlretrieve(backup_url, filename)

    ext = Path(filename).suffixes[-1][1:]

    if ext in numpy_ext:
        return np.load(filename, **kwargs)
    elif ext in pandas_ext:
        return pd.read_csv(filename, **kwargs)
    elif ext in adata_ext:
        return anndata.read(filename, **kwargs)
    else:
        raise ValueError('"{}" does not end on a valid extension.\n'
                         'Please, provide one of the available extensions.\n{}\n'
                         .format(filename, numpy_ext | pandas_ext))


def load_celeba(file_path, attr_path,
                gender='Male', source_attr="Black_Hair", target_attr="Blond_Hair",
                max_n_images=None,
                save=True, restore=True,
                img_resize=64,
                verbose=True,
                balanced=True,
                preprocess=True):
    data_path = os.path.dirname(file_path)

    if restore and os.path.exists(os.path.join(data_path, "source_images.npy")):
        source_images = np.load(os.path.join(data_path, "source_images.npy"))
        target_images = np.load(os.path.join(data_path, "target_images.npy"))
        return source_images, target_images
    elif restore:
        raise Exception("npy files does not exist!")

    def load_attr_list(file_path):
        indices = []
        attributes = []
        with open(file_path) as f:
            lines = f.read().splitlines()
            columns = lines[1].split(" ")
            columns.remove('')
            for i in range(2, len(lines)):
                elements = lines[i].split()
                indices.append(elements[0])
                attributes.append(list(map(int, elements[1:])))
        attr_df = pd.DataFrame(attributes)
        attr_df.index = indices
        attr_df.columns = columns
        if gender is not None:
            attr_df = attr_df[attr_df[gender] == 1]
        if verbose:
            print(attr_df.shape[0])
        if source_attr != target_attr:
            attr_df = attr_df.loc[((attr_df[source_attr] == 1) & (attr_df[target_attr] == -1) | (
                    attr_df[source_attr] == -1) & (attr_df[target_attr] == 1))]
        return attr_df

    images = []
    zfile = zipfile.ZipFile(file_path)
    counter = 0
    attr_df = load_attr_list(attr_path)
    print(len(attr_df.index.tolist()))
    indices = []
    for filename in attr_df.index.tolist():
        ifile = zfile.open(os.path.join("img_align_celeba/", filename))
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

    if source_attr != target_attr:
        source_images = images_df[attr_df[source_attr] == 1]
        target_images = images_df[attr_df[target_attr] == 1]
    else:
        source_images = images_df[attr_df[source_attr] == -1]
        target_images = images_df[attr_df[source_attr] == 1]

    source_images = np.reshape(source_images.values, (-1, img_resize, img_resize, 3))
    target_images = np.reshape(target_images.values, (-1, img_resize, img_resize, 3))

    source_images = np.array(source_images, dtype=np.float32)
    target_images = np.array(target_images, dtype=np.float32)
    if preprocess:
        source_images /= 255.0
        target_images /= 255.0

    if verbose:
        print("Without Balancing: ", source_images.shape, target_images.shape)
    if balanced:
        min_size = min(source_images.shape[0], target_images.shape[0])

        source_indices = np.random.choice(source_images.shape[0], min_size, replace=False)
        source_images = source_images[source_indices]

        target_indices = np.random.choice(target_images.shape[0], min_size, replace=False)
        target_images = target_images[target_indices]

    if verbose:
        print(source_images.shape, target_images.shape)
    if save:
        np.save(arr=source_images, file=os.path.join(data_path, f"source_images.npy"), allow_pickle=True)
        np.save(arr=target_images, file=os.path.join(data_path, f"target_images.npy"), allow_pickle=True)
    return source_images, target_images


def resize_image(images, img_size):
    images_list = []
    for i in range(images.shape[0]):
        image = cv2.resize(images[i], (img_size, img_size), cv2.INTER_NEAREST)
        images_list.append(image)
    return np.array(images_list)
