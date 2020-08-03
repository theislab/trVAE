import argparse
import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import trvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

FASHION_MNIST_CLASS_DICT = {
    0: "T-shirt or top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

DATASETS = {
    "CelebA": {"name": 'celeba', "gender": "Male", 'attribute': "Smiling", 'source_key': -1, "target_key": 1,
               "width": 64, 'height': 64, "n_channels": 3},

    "MNIST": {"name": 'mnist', "source_key": 1, "target_key": 7,
              "train_digits": [], "test_digits": [],
              "width": 28, 'height': 28, "n_channels": 1},

    "ThinMNIST": {"name": 'thin_mnist', "source_key": "normal", "target_key": "thin",
                  'train_digits': [1, 3, 6, 7], 'test_digits': [0, 2, 4, 5, 8, 9],
                  "width": 28, 'height': 28,
                  "n_channels": 1},

    "ThickMNIST": {"name": 'thick_mnist', "source_key": "normal", "target_key": "thick",
                   'train_digits': [1, 3, 6, 7], 'test_digits': [0, 2, 4, 5, 8, 9],
                   "width": 28, 'height': 28,
                   "n_channels": 1},

    "FashionMNIST": {"name": "fashion_mnist", "source_key": FASHION_MNIST_CLASS_DICT[0],
                     "target_key": FASHION_MNIST_CLASS_DICT[1],
                     "width": 28, 'height': 28, "n_channels": 1},

    # "Horse2Zebra": {"name": "h2z", "source_key": "horse", "target_key": "zebra", "size": 256, "n_channels": 3,
    #                 "resize": 64},
    # "Apple2Orange": {"name": "a2o", "source_key": "apple", "target_key": "orange", "size": 256, "n_channels": 3,
    #                  "resize": 64}
}


def train_network(data_dict=None,
                  z_dim=100,
                  mmd_dimension=256,
                  alpha=0.001,
                  beta=100,
                  gamma=1.0,
                  kernel='multi-scale-rbf',
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  arch_style=1,
                  preprocess=True,
                  learning_rate=0.001,
                  gpus=1,
                  max_size=50000,
                  early_stopping_limit=50,
                  ):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_width = data_dict.get("width", None)
    img_height = data_dict.get("height", None)
    n_channels = data_dict.get("n_channels", None)
    train_digits = data_dict.get("train_digits", None)
    test_digits = data_dict.get("test_digits", None)
    attribute = data_dict.get('attribute', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = trvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             landmark_path="../data/celeba/list_landmarks_align_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=max_size,
                                             img_width=img_width,
                                             img_height=img_height,
                                             restore=True,
                                             save=True)

        if sparse.issparse(data.X):
            data.X = data.X.A

        source_images = data.copy()[data.obs['condition'] == source_key].X
        target_images = data.copy()[data.obs['condition'] == target_key].X

        source_images = np.reshape(source_images, (-1, img_width, img_height, n_channels))
        target_images = np.reshape(target_images, (-1, img_width, img_height, n_channels))

        if preprocess:
            source_images /= 255.0
            target_images /= 255.0
    else:
        data = sc.read(f"../data/{data_name}/{data_name}.h5ad")

        source_images = data.copy()[data.obs["condition"] == source_key].X
        target_images = data.copy()[data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_width, img_height, n_channels))
        target_images = np.reshape(target_images, (-1, img_width, img_height, n_channels))

        if preprocess:
            source_images /= 255.0
            target_images /= 255.0

    source_labels = np.zeros(shape=source_images.shape[0])
    target_labels = np.ones(shape=target_images.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)

    train_images = np.concatenate([source_images, target_images], axis=0)
    train_images = np.reshape(train_images, (-1, np.prod(source_images.shape[1:])))
    if data_name.__contains__('mnist'):
        preprocessed_data = anndata.AnnData(X=train_images)
        preprocessed_data.obs["condition"] = train_labels
        preprocessed_data.obs['labels'] = data.obs['labels'].values
        data = preprocessed_data.copy()
    else:
        preprocessed_data = anndata.AnnData(X=train_images)
        preprocessed_data.obs['condition'] = train_labels
        if data.obs.columns.__contains__('labels'):
            preprocessed_data.obs['labels'] = data.obs['condition'].values
        data = preprocessed_data.copy()

    train_size = int(data.shape[0] * 0.85)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    data_train = data[train_idx, :]
    data_valid = data[test_idx, :]
    print(data_train.shape, data_valid.shape)

    if train_digits is not None:
        train_data = data_train.copy()[
            ~((data_train.obs['labels'].isin(test_digits)) & (data_train.obs['condition'] == 1))]
        valid_data = data_valid.copy()[
            ~((data_valid.obs['labels'].isin(test_digits)) & (data_valid.obs['condition'] == 1))]
    elif data_name == "celeba":
        train_data = data_train.copy()[
            ~((data_train.obs['labels'] == -1) & (data_train.obs['condition'] == target_key))]
        valid_data = data_valid.copy()[
            ~((data_valid.obs['labels'] == -1) & (data_valid.obs['condition'] == target_key))]
    else:
        train_data = data_train.copy()
        valid_data = data_valid.copy()

    network = trvae.archs.DCtrVAE(x_dimension=source_images.shape[1:],
                                  z_dimension=z_dim,
                                  mmd_dimension=mmd_dimension,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  kernel=kernel,
                                  arch_style=arch_style,
                                  train_with_fake_labels=False,
                                  learning_rate=learning_rate,
                                  model_path=f"../models/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/",
                                  gpus=gpus,
                                  dropout_rate=dropout_rate)

    print(train_data.shape, valid_data.shape)
    network.train(train_data,
                  use_validation=True,
                  valid_adata=valid_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=early_stopping_limit,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


def evaluate_network(data_dict=None, z_dim=100, n_files=5, k=5, arch_style=1, preprocess=True, max_size=80000):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_width = data_dict.get("width", None)
    img_height = data_dict.get("height", None)
    n_channels = data_dict.get('n_channels', None)
    train_digits = data_dict.get('train_digits', None)
    test_digits = data_dict.get('test_digits', None)
    attribute = data_dict.get('attribute', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = trvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             landmark_path="../data/celeba/list_landmarks_align_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=max_size,
                                             img_width=img_width,
                                             img_height=img_height,
                                             restore=True,
                                             save=False)

        valid_data = data.copy()[data.obs['labels'] == -1]  # get females (Male = -1)
        train_data = data.copy()[data.obs['labels'] == +1]  # get males (Male = 1)

        if sparse.issparse(valid_data.X):
            valid_data.X = valid_data.X.A

        source_images_train = train_data[train_data.obs["condition"] == source_key].X
        source_images_valid = valid_data[valid_data.obs["condition"] == source_key].X

        source_images_train = np.reshape(source_images_train, (-1, img_width, img_height, n_channels))
        source_images_valid = np.reshape(source_images_valid, (-1, img_width, img_height, n_channels))

        if preprocess:
            source_images_train /= 255.0
            source_images_valid /= 255.0
    else:
        data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        if train_digits is not None:
            train_data = data[data.obs['labels'].isin(train_digits)]
            valid_data = data[data.obs['labels'].isin(test_digits)]
        else:
            train_data = data.copy()
            valid_data = data.copy()

        source_images_train = train_data[train_data.obs["condition"] == source_key].X
        target_images_train = train_data[train_data.obs["condition"] == target_key].X

        source_images_train = np.reshape(source_images_train, (-1, img_width, img_height, n_channels))
        target_images_train = np.reshape(target_images_train, (-1, img_width, img_height, n_channels))

        source_images_valid = valid_data[valid_data.obs["condition"] == source_key].X
        target_images_valid = valid_data[valid_data.obs["condition"] == target_key].X

        source_images_valid = np.reshape(source_images_valid, (-1, img_width, img_height, n_channels))
        target_images_valid = np.reshape(target_images_valid, (-1, img_width, img_height, n_channels))

        if preprocess:
            source_images_train /= 255.0
            source_images_valid /= 255.0

            target_images_train /= 255.0
            target_images_valid /= 255.0

    image_shape = (img_width, img_height, n_channels)

    source_images_train = np.reshape(source_images_train, (-1, np.prod(image_shape)))
    source_images_valid = np.reshape(source_images_valid, (-1, np.prod(image_shape)))

    source_data_train = anndata.AnnData(X=source_images_train)
    source_data_valid = anndata.AnnData(X=source_images_valid)

    network = trvae.DCtrVAE(x_dimension=image_shape,
                            z_dimension=z_dim,
                            arch_style=arch_style,
                            model_path=f"../models/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/")

    network.restore_model()

    results_path_train = f"../results/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/{source_key} to {target_key}/train/"
    results_path_valid = f"../results/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/{source_key} to {target_key}/valid/"
    os.makedirs(results_path_train, exist_ok=True)
    os.makedirs(results_path_valid, exist_ok=True)

    if sparse.issparse(valid_data.X):
        valid_data.X = valid_data.X.A
    if test_digits is not None:
        k = len(test_digits)
    for j in range(n_files):
        if test_digits is not None:
            source_sample_train = []
            source_sample_valid = []

            target_sample_train = []
            target_sample_valid = []

            for digit in test_digits:
                source_images_digit_valid = valid_data[
                    (valid_data.obs['labels'] == digit) & (valid_data.obs['condition'] == source_key)]
                target_images_digit_valid = valid_data[
                    (valid_data.obs['labels'] == digit) & (valid_data.obs['condition'] == target_key)]
                if j == 0:
                    source_images_digit_valid.X /= 255.0
                random_samples = np.random.choice(source_images_digit_valid.shape[0], 1, replace=False)

                source_sample_valid.append(source_images_digit_valid.X[random_samples])
                target_sample_valid.append(target_images_digit_valid.X[random_samples])

            for digit in train_digits:
                source_images_digit_train = train_data[
                    (train_data.obs['labels'] == digit) & (train_data.obs['condition'] == source_key)]
                target_images_digit_train = train_data[
                    (train_data.obs['labels'] == digit) & (train_data.obs['condition'] == target_key)]
                if j == 0:
                    source_images_digit_train.X /= 255.0
                random_samples = np.random.choice(source_images_digit_train.shape[0], 1, replace=False)

                source_sample_train.append(source_images_digit_train.X[random_samples])
                target_sample_train.append(target_images_digit_train.X[random_samples])
        else:
            random_samples_train = np.random.choice(source_data_train.shape[0], k, replace=False)
            random_samples_valid = np.random.choice(source_data_valid.shape[0], k, replace=False)
            source_sample_train = source_data_train.X[random_samples_train]
            source_sample_valid = source_data_valid.X[random_samples_valid]

        source_sample_train = np.array(source_sample_train)
        source_sample_valid = np.array(source_sample_valid)
        # if data_name.__contains__("mnist"):
        #     target_sample = np.array(target_sample)
        #     target_sample_reshaped = np.reshape(target_sample, (-1, *image_shape))

        source_sample_train = np.reshape(source_sample_train, (-1, np.prod(image_shape)))
        source_sample_train_reshaped = np.reshape(source_sample_train, (-1, *image_shape))
        if data_name.__contains__("mnist"):
            target_sample_train = np.reshape(target_sample_train, (-1, np.prod(image_shape)))
            target_sample_train_reshaped = np.reshape(target_sample_train, (-1, *image_shape))
            target_sample_valid = np.reshape(target_sample_valid, (-1, np.prod(image_shape)))
            target_sample_valid_reshaped = np.reshape(target_sample_valid, (-1, *image_shape))

        source_sample_valid = np.reshape(source_sample_valid, (-1, np.prod(image_shape)))
        source_sample_valid_reshaped = np.reshape(source_sample_valid, (-1, *image_shape))

        source_sample_train = anndata.AnnData(X=source_sample_train)
        source_sample_valid = anndata.AnnData(X=source_sample_valid)

        pred_sample_train = network.predict(adata=source_sample_train,
                                            encoder_labels=np.zeros((k, 1)),
                                            decoder_labels=np.ones((k, 1)))
        pred_sample_train = np.reshape(pred_sample_train, newshape=(-1, *image_shape))

        pred_sample_valid = network.predict(adata=source_sample_valid,
                                            encoder_labels=np.zeros((k, 1)),
                                            decoder_labels=np.ones((k, 1)))
        pred_sample_valid = np.reshape(pred_sample_valid, newshape=(-1, *image_shape))

        print(source_sample_train.shape, source_sample_train_reshaped.shape, pred_sample_train.shape)

        plt.close("all")
        if train_digits is not None:
            k = len(train_digits)
        if data_name.__contains__("mnist"):
            fig, ax = plt.subplots(len(train_digits), 3, figsize=(k * 1, 6))
        else:
            fig, ax = plt.subplots(k, 2, figsize=(k * 1, 6))
        for i in range(k):
            ax[i, 0].axis('off')
            if source_sample_train_reshaped.shape[-1] > 1:
                ax[i, 0].imshow(source_sample_train_reshaped[i])
            else:
                ax[i, 0].imshow(source_sample_train_reshaped[i, :, :, 0], cmap='Greys')
            ax[i, 1].axis('off')
            if data_name.__contains__("mnist"):
                ax[i, 2].axis('off')
            # if i == 0:
            #     if data_name == "celeba":
            #         ax[i, 0].set_title(f"without {data_dict['attribute']}")
            #         ax[i, 1].set_title(f"with {data_dict['attribute']}")
            #     elif data_name.__contains__("mnist"):
            #         ax[i, 0].set_title(f"Source")
            #         ax[i, 1].set_title(f"Target (Ground Truth)")
            #         ax[i, 2].set_title(f"Target (Predicted)")
            #     else:
            #         ax[i, 0].set_title(f"{source_key}")
            #         ax[i, 1].set_title(f"{target_key}")

            if pred_sample_train.shape[-1] > 1:
                ax[i, 1].imshow(pred_sample_train[i])
            else:
                ax[i, 1].imshow(target_sample_train_reshaped[i, :, :, 0], cmap='Greys')
                ax[i, 2].imshow(pred_sample_train[i, :, :, 0], cmap='Greys')
            # if data_name.__contains__("mnist"):
            #     ax[i, 2].imshow(target_sample_reshaped[i, :, :, 0], cmap='Greys')
        plt.savefig(os.path.join(results_path_train, f"sample_images_{j}.pdf"))

        print(source_sample_valid.shape, source_sample_valid_reshaped.shape, pred_sample_valid.shape)

        plt.close("all")
        if test_digits is not None:
            k = len(test_digits)
        if data_name.__contains__("mnist"):
            fig, ax = plt.subplots(k, 3, figsize=(k * 1, 6))
        else:
            fig, ax = plt.subplots(k, 2, figsize=(k * 1, 6))
        for i in range(k):
            ax[i, 0].axis('off')
            if source_sample_valid_reshaped.shape[-1] > 1:
                ax[i, 0].imshow(source_sample_valid_reshaped[i])
            else:
                ax[i, 0].imshow(source_sample_valid_reshaped[i, :, :, 0], cmap='Greys')
            ax[i, 1].axis('off')
            if data_name.__contains__("mnist"):
                ax[i, 2].axis('off')
            # if i == 0:
            #     if data_name == "celeba":
            #         ax[i, 0].set_title(f"without {data_dict['attribute']}")
            #         ax[i, 1].set_title(f"with {data_dict['attribute']}")
            #     elif data_name.__contains__("mnist"):
            #         ax[i, 0].set_title(f"Source")
            #         ax[i, 1].set_title(f"Target (Ground Truth)")
            #         ax[i, 2].set_title(f"Target (Predicted)")
            #     else:
            #         ax[i, 0].set_title(f"{source_key}")
            #         ax[i, 1].set_title(f"{target_key}")

            if pred_sample_valid.shape[-1] > 1:
                ax[i, 1].imshow(pred_sample_valid[i])
            else:
                ax[i, 1].imshow(target_sample_valid_reshaped[i, :, :, 0], cmap='Greys')
                ax[i, 2].imshow(pred_sample_valid[i, :, :, 0], cmap='Greys')
            # if data_name.__contains__("mnist"):
            #     ax[i, 2].imshow(target_sample_reshaped[i, :, :, 0], cmap='Greys')
        plt.savefig(os.path.join(results_path_valid, f"./sample_images_{j}.pdf"))


def visualize_trained_network_results(data_dict, z_dim=100, arch_style=1, preprocess=True, max_size=80000):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_width = data_dict.get('width', None)
    img_height = data_dict.get('height', None)
    n_channels = data_dict.get('n_channels', None)
    train_digits = data_dict.get('train_digits', None)
    test_digits = data_dict.get('test_digits', None)
    attribute = data_dict.get('attribute', None)

    path_to_save = f"../results/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/{source_key} to {target_key}/UMAPs/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = trvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             landmark_path="../data/celeba/list_landmarks_align_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=max_size,
                                             img_width=img_width,
                                             img_height=img_height,
                                             restore=True,
                                             save=False)

        if sparse.issparse(data.X):
            data.X = data.X.A

        train_images = data.X
        train_data = anndata.AnnData(X=data)
        train_data.obs['condition'] = data.obs['condition'].values
        train_data.obs.loc[train_data.obs['condition'] == 1, 'condition'] = f'with {attribute}'
        train_data.obs.loc[train_data.obs['condition'] == -1, 'condition'] = f'without {attribute}'

        train_data.obs['labels'] = data.obs['labels'].values
        train_data.obs.loc[train_data.obs['labels'] == 1, 'labels'] = f'Male'
        train_data.obs.loc[train_data.obs['labels'] == -1, 'labels'] = f'Female'

        if preprocess:
            train_images /= 255.0
    else:
        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        train_images = np.reshape(train_data.X, (-1, img_width, img_height, n_channels))

        if preprocess:
            train_images /= 255.0

    train_labels, _ = trvae.label_encoder(train_data)
    fake_labels = np.ones(train_labels.shape)

    network = trvae.DCtrVAE(x_dimension=(img_width, img_height, n_channels),
                            z_dimension=z_dim,
                            arch_style=arch_style,
                            model_path=f"../models/RCCVAE/{data_name}-{img_width}x{img_height}-{preprocess}/{arch_style}-{z_dim}/", )

    network.restore_model()

    train_data_feed = np.reshape(train_images, (-1, img_width, img_height, n_channels))

    latent_with_true_labels = network.to_z_latent(train_data_feed, train_labels)
    latent_with_fake_labels = network.to_z_latent(train_data_feed, fake_labels)
    mmd_latent_with_true_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=False)
    mmd_latent_with_fake_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=True)

    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs['condition'] = pd.Categorical(train_data.obs['condition'].values)

    latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
    latent_with_fake_labels.obs['condition'] = pd.Categorical(train_data.obs['condition'].values)

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs['condition'] = train_data.obs['condition'].values

    mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
    mmd_latent_with_fake_labels.obs['condition'] = train_data.obs['condition'].values

    if data_name.__contains__("mnist") or data_name == "celeba":
        latent_with_true_labels.obs['labels'] = pd.Categorical(train_data.obs['labels'].values)
        latent_with_fake_labels.obs['labels'] = pd.Categorical(train_data.obs['labels'].values)
        mmd_latent_with_true_labels.obs['labels'] = pd.Categorical(train_data.obs['labels'].values)
        mmd_latent_with_fake_labels.obs['labels'] = pd.Categorical(train_data.obs['labels'].values)

        color = ['condition', 'labels']
    else:
        color = ['condition']

    if train_digits is not None:
        train_data.obs.loc[(train_data.obs['condition'] == source_key) & (
            train_data.obs['labels'].isin(train_digits)), 'type'] = 'training'
        train_data.obs.loc[
            (train_data.obs['condition'] == source_key) & (
                train_data.obs['labels'].isin(test_digits)), 'type'] = 'training'
        train_data.obs.loc[(train_data.obs['condition'] == target_key) & (
            train_data.obs['labels'].isin(train_digits)), 'type'] = 'training'
        train_data.obs.loc[
            (train_data.obs['condition'] == target_key) & (
                train_data.obs['labels'].isin(test_digits)), 'type'] = 'heldout'

    sc.pp.neighbors(train_data)
    sc.tl.umap(train_data)
    sc.pl.umap(train_data, color=color,
               save=f'_{data_name}_train_data.png',
               show=False,
               wspace=0.5)

    if train_digits is not None:
        sc.tl.umap(train_data)
        sc.pl.umap(train_data, color=['type', 'labels'],
                   save=f'_{data_name}_data_type.png',
                   show=False)

    sc.pp.neighbors(latent_with_true_labels)
    sc.tl.umap(latent_with_true_labels)
    sc.pl.umap(latent_with_true_labels, color=color,
               save=f"_{data_name}_latent_with_true_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(latent_with_fake_labels)
    sc.tl.umap(latent_with_fake_labels)
    sc.pl.umap(latent_with_fake_labels, color=color,
               save=f"_{data_name}_latent_with_fake_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(mmd_latent_with_true_labels)
    sc.tl.umap(mmd_latent_with_true_labels)
    sc.pl.umap(mmd_latent_with_true_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_true_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(mmd_latent_with_fake_labels)
    sc.tl.umap(mmd_latent_with_fake_labels)
    sc.pl.umap(mmd_latent_with_fake_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_fake_labels.png",
               wspace=0.5,
               show=False)

    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=20, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-m', '--mmd_dimension', type=int, default=128, required=False,
                                 help='MMD Layer dimension')
    arguments_group.add_argument('-a', '--alpha', type=float, default=0.005, required=False,
                                 help='Alpha coeff in loss term')
    arguments_group.add_argument('-b', '--beta', type=float, default=100, required=False,
                                 help='Beta coeff in loss term')
    arguments_group.add_argument('-k', '--kernel', type=str, default='multi-scale-rbf', required=False,
                                 help='Kernel type')
    arguments_group.add_argument('-n', '--n_epochs', type=int, default=5000, required=False,
                                 help='Maximum Number of epochs for training')
    arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                                 help='Batch Size')
    arguments_group.add_argument('-s', '--arch_style', type=int, default=1, required=False,
                                 help='Model Architecture Style')
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.4, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-w', '--width', type=int, default=0, required=False,
                                 help='Image Width to be resize')
    arguments_group.add_argument('-e', '--height', type=int, default=0, required=False,
                                 help='Image Height to be resize')
    arguments_group.add_argument('-p', '--preprocess', type=int, default=True, required=False,
                                 help='do preprocess images')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-g', '--gpus', type=int, default=1, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-x', '--max_size', type=int, default=50000, required=False,
                                 help='Max Size for CelebA')
    arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                                 help='do train the network')
    arguments_group.add_argument('-y', '--early_stopping_limit', type=int, default=50, required=False,
                                 help='do train the network')
    arguments_group.add_argument('-f', '--gamma', type=float, default=1.0, required=False,
                                 help='do train the network')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    if args['width'] > 0 and args['height'] > 0:
        data_dict['width'] = args['width']
        data_dict['height'] = args['height']

    if args['preprocess'] == 0:
        args['preprocess'] = False
    else:
        args['preprocess'] = True

    if args['max_size'] == 0:
        args['max_size'] = None

    del args['data']
    del args['width']
    del args['height']
    if args['do_train'] > 0:
        del args['do_train']
        train_network(data_dict=data_dict, **args)
    evaluate_network(data_dict,
                     z_dim=args['z_dim'],
                     n_files=30,
                     arch_style=args['arch_style'],
                     max_size=args['max_size'],
                     k=4)
    # visualize_trained_network_results(data_dict,
    #                                   z_dim=args['z_dim'],
    #                                   arch_style=args['arch_style'],
    #                                   max_size=args['max_size'],
    #                                   preprocess=args['preprocess'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
