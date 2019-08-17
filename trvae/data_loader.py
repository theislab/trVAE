import os
import tarfile
import zipfile

import anndata
import cv2
import keras
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


def prepare_and_load_celeba(file_path, attr_path, landmark_path,
                            gender='Male', attribute='Smiling',
                            max_n_images=None,
                            restore=True,
                            save=True,
                            img_width=64, img_height=78,
                            verbose=True):
    data_path = os.path.dirname(file_path)
    zip_filename = os.path.basename(file_path).split(".")[0]
    if restore and os.path.exists(
            os.path.join(data_path, f"celeba_{attribute}_{img_width}x{img_height}_{max_n_images}.h5ad")):
        return sc.read(os.path.join(data_path, f"celeba_{attribute}_{img_width}x{img_height}_{max_n_images}.h5ad"))

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
        if verbose:
            print(attr_df.shape[0])
        return attr_df

    def load_landmark_list(file_path):
        indices = []
        landmarks = []
        with open(file_path) as f:
            lines = f.read().splitlines()
            columns = lines[1].split(" ")
            for i in range(2, len(lines)):
                elements = lines[i].split()
                indices.append(elements[0])
                landmarks.append(list(map(int, elements[1:])))
        landmarks_df = pd.DataFrame(landmarks)
        landmarks_df.index = indices
        landmarks_df.columns = columns
        print(landmarks_df.shape[0])
        return landmarks_df

    images = []
    zfile = zipfile.ZipFile(file_path)
    counter = 0
    attr_df = load_attr_list(attr_path)
    landmarks = load_landmark_list(landmark_path)
    landmarks = landmarks[abs(landmarks['lefteye_x'] - landmarks['righteye_x']) > 30]
    landmarks = landmarks[abs(landmarks['lefteye_x'] - landmarks['nose_x']) > 15]
    landmarks = landmarks[abs(landmarks['righteye_x'] - landmarks['nose_x']) > 15]
    landmarks.head()
    attr_df = attr_df.loc[landmarks.index]
    print("# of images after preprocessing: ", attr_df.shape[0])

    indices = []
    for filename in attr_df.index.tolist():
        ifile = zfile.open(os.path.join(f"{zip_filename}/", filename))
        image = Image.open(ifile)
        image_landmarks = landmarks.loc[filename]
        most_left_x = max(0, min(image_landmarks['lefteye_x'], image_landmarks['leftmouth_x']) - 15)
        most_right_x = min(178, min(image_landmarks['righteye_x'], image_landmarks['rightmouth_x']) + 15)

        most_up_y = max(0, image_landmarks['lefteye_y'] - 35)
        most_down_y = min(218, image_landmarks['rightmouth_y'] + 25)

        image_cropped = image.crop((most_left_x, most_up_y, most_right_x, most_down_y))
        image_cropped = image_cropped.resize((img_width, img_height), Image.NEAREST)
        image = image_cropped

        image = np.reshape(image, (img_width, img_height, 3))

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
        attr_df = attr_df.loc[images_df.index]
        print(data.shape, attr_df.shape)
        data.obs['labels'] = attr_df[gender].values
        data.obs['condition'] = attr_df[attribute].values
        sc.write(filename=os.path.join(data_path, f"celeba_{attribute}_{img_width}x{img_height}_{max_n_images}.h5ad"),
                 adata=data)
    return data


def prepare_and_load_edge2shoe(file_path,
                               restore=True, save=True,
                               img_width=64, img_height=64,
                               verbose=True):
    data_path = os.path.dirname(file_path)
    if restore and os.path.exists(os.path.join(data_path, f"edges2shoes_{img_width}x{img_height}.h5ad")):
        return sc.read(os.path.join(data_path, f"edges2shoes_{img_width}x{img_height}.h5ad"))

    tar = tarfile.open(file_path)
    images, edges = [], []

    counter = 0
    for member in tar.getmembers():
        if member.name.endswith(".jpg"):
            f = tar.extractfile(member)
            image = Image.open(f)

            edge, image = image.crop((0, 0, 256, 256)), image.crop((256, 0, 512, 256))

            edge = edge.resize((64, 64), Image.BICUBIC)
            image = image.resize((64, 64), Image.NEAREST)

            edge = np.array(edge)
            image = np.array(image)

            images.append(image)
            edges.append(edge)

            counter += 1
            if verbose and counter % 1000 == 0:
                print(counter)
    images = np.array(images)
    edges = np.array(edges)

    images = images.reshape(-1, np.prod(images.shape[1:]))
    edges = edges.reshape(-1, np.prod(edges.shape[1:]))

    data = np.concatenate([images, edges], axis=0)

    if save:
        data = anndata.AnnData(X=data)
        data.obs['id'] = np.concatenate([np.arange(images.shape[0]), np.arange(images.shape[0])])
        data.obs['condition'] = ['shoe'] * images.shape[0] + ['edge'] * images.shape[0]
        sc.write(filename=os.path.join(data_path, f"edges2shoes_{img_width}x{img_height}.h5ad"), adata=data)
    return data


def resize_image(images, img_width, img_height):
    images_list = []
    for i in range(images.shape[0]):
        image = cv2.resize(images[i], (img_width, img_height), cv2.INTER_NEAREST)
        images_list.append(image)
    return np.array(images_list)


class PairedDataSequence(keras.utils.Sequence):
    def __init__(self, image_paths, batch_size):
        self.image_paths = image_paths
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        edges, images = [], []
        batch_image_paths = self.image_paths[idx:idx + self.batch_size]

        for image_path in batch_image_paths:
            with Image.open(image_path) as image:
                edge = np.array(image.crop((0, 0, 256, 256)).resize((64, 64), Image.BICUBIC))
                image = np.array(image.crop((256, 0, 512, 256)).resize((64, 64), Image.NEAREST))
                edges.append(edge)
                images.append(image)

        edges = np.array(edges)
        images = np.array(images)

        edges = edges.astype(np.float32)
        images = images.astype(np.float32)

        # Pre-processing
        edges /= 255.0
        images /= 255.0

        x = np.concatenate([edges, edges, images, images], axis=0)
        y = np.concatenate([edges, images, images, edges], axis=0)
        encoder_labels_feed = np.concatenate([np.zeros(edges.shape[0]), np.zeros(edges.shape[0]),
                                              np.ones(images.shape[0]), np.ones(images.shape[0])])

        decoder_labels_feed = np.concatenate([np.zeros(edges.shape[0]), np.ones(edges.shape[0]),
                                              np.ones(images.shape[0]), np.zeros(images.shape[0])])
        x_feed = [x, encoder_labels_feed, decoder_labels_feed]
        y_feed = [y, encoder_labels_feed]
        return x_feed, y_feed
