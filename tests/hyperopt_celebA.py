from __future__ import print_function

import argparse
import os

import anndata
import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from matplotlib import pyplot as plt
from scipy import sparse

import trvae


def data():
    DATASETS = {
        "CelebA": {"name": 'celeba', "gender": "Male", 'attribute': "Smiling", 'source_key': -1, "target_key": 1,
                   "width": 64, 'height': 64, "n_channels": 3},

    }
    data_key = "CelebA"
    data_dict = DATASETS[data_key]
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_width = data_dict.get("width", None)
    img_height = data_dict.get("height", None)
    n_channels = data_dict.get("n_channels", None)
    attribute = data_dict.get('attribute', None)
    gender = data_dict.get('gender', None)

    data = trvae.prepare_and_load_celeba(file_path="./data/celeba/img_align_celeba.zip",
                                         attr_path="./data/celeba/list_attr_celeba.txt",
                                         landmark_path="./data/celeba/list_landmarks_align_celeba.txt",
                                         gender=gender,
                                         attribute=attribute,
                                         max_n_images=50000,
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

    source_images /= 255.0
    target_images /= 255.0

    source_labels = np.zeros(shape=source_images.shape[0])
    target_labels = np.ones(shape=target_images.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)

    train_images = np.concatenate([source_images, target_images], axis=0)
    train_images = np.reshape(train_images, (-1, np.prod(source_images.shape[1:])))

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

    train_data = data_train.copy()[
        ~((data_train.obs['labels'] == -1) & (data_train.obs['condition'] == target_key))]
    valid_data = data_valid.copy()[
        ~((data_valid.obs['labels'] == -1) & (data_valid.obs['condition'] == target_key))]

    return train_data, valid_data, data_name


def create_model(train_data, valid_data, data_name):
    z_dim_choices = {{choice([20, 40, 50, 60, 80, 100])}}
    mmd_dim_choices = {{choice([64, 128, 256])}}

    alpha_choices = {{choice([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
    beta_choices = {{choice([1, 5, 10, 50, 100, 500, 1000])}}
    # gamma_choices = {{choice([0.001, 0.01, 0.1, 1, 10.0])}}
    batch_size_choices = {{choice([256, 512, 1024])}}
    dropout_rate_choices = {{choice([0.1, 0.2, 0.5, 0.75])}}

    network = trvae.DCtrVAE(x_dimension=(64, 64, 3),
                            z_dimension=z_dim_choices,
                            mmd_dimension=mmd_dim_choices,
                            alpha=alpha_choices,
                            beta=beta_choices,
                            gamma=0,
                            kernel='rbf',
                            arch_style=3,
                            train_with_fake_labels=False,
                            learning_rate=0.001,
                            model_path=f"./models/RCCVAE/hyperopt/{data_name}-{64}x{64}-{True}/{3}-{z_dim_choices}/",
                            gpus=4,
                            dropout_rate=dropout_rate_choices)

    history = network.train(train_data,
                            use_validation=True,
                            valid_adata=valid_data,
                            n_epochs=10000,
                            batch_size=batch_size_choices,
                            verbose=2,
                            early_stop_limit=200,
                            shuffle=True,
                            save=True)

    print(f'Best Reconstruction Loss of model: ({history.history["val_kl_reconstruction_loss"][0]})')
    return {'loss': history.history["val_kl_reconstruction_loss"][0], 'status': STATUS_OK}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-n', '--max_evals', type=int, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())
    best_run, best_network = optim.minimize(model=create_model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=args['max_evals'],
                                            trials=Trials())
    print("All Done!")
    print(best_run)
