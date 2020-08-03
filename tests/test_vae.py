import argparse
import os

import numpy as np
import scanpy as sc
from scipy import sparse

import trvae
from matplotlib import pyplot as plt

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")


DATASETS = {
    "atac": {"name": 'atac',
             "matrix_file": "atac_matrix.binary.qc_filtered.mtx.gz",
             "metadata": "cell_metadata.txt",
             'cell_type': 'Cusanovich_label',
             'spec_cell_types': [],
             },

}


def train_network(data_dict=None,
                  z_dim=100,
                  subsample=None,
                  alpha=0.001,
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  gpus=1,
                  verbose=2,
                  arch_style=1,
                  ):
    data_name = data_dict['name']
    metadata_path = data_dict['metadata']
    cell_type_key = data_dict['cell_type']

    train_data = sc.read(f"../data/{data_name}/anna/processed_adata_Cusanovich_brain_May29_2019_5000.h5ad")
    train_data.X += abs(train_data.X.min())
    if subsample is not None:
        train_data = train_data[:subsample]

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type is not []:
        cell_types = spec_cell_type

    train_size = int(train_data.shape[0] * 0.85)
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:]

    net_train_data = train_data.copy()[train_idx, :]
    net_valid_data = train_data.copy()[valid_idx, :]

    network = trvae.VAE(x_dimension=net_train_data.shape[1],
                        z_dimension=z_dim,
                        alpha=alpha,
                        gpus=gpus,
                        learning_rate=learning_rate,
                        model_path=f"../models/VAE/{data_name}-{arch_style}/{z_dim}/",
                        arch_style=arch_style,
                        dropout_rate=dropout_rate)

    network.train(net_train_data,
                  use_validation=True,
                  valid_adata=net_valid_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=verbose,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print(f"Model for {data_name} has been trained")


def visualize_trained_network_results(data_dict, z_dim=100, subsample=None, arch_style=1):
    plt.close("all")
    data_name = data_dict['name']
    metadata_path = data_dict['metadata']
    cell_type_key = data_dict['cell_type']
    spec_cell_type = data_dict.get("spec_cell_types", None)

    data = sc.read(f"../data/{data_name}/anna/processed_adata_Cusanovich_brain_May29_2019_5000.h5ad")
    data.X += abs(data.X.min())
    if subsample is not None:
        data = data[:subsample]
    cell_types = data.obs[cell_type_key].unique().tolist()

    path_to_save = f"../results/VAE/{data_name}/{arch_style}-{z_dim}/Visualizations/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    train_data = data.copy()

    network = trvae.VAE(x_dimension=data.shape[1],
                        z_dimension=z_dim,
                        arch_style=arch_style,
                        model_path=f"../models/VAE/{data_name}-{arch_style}/{z_dim}/", )

    network.restore_model()

    if sparse.issparse(data.X):
        data.X = data.X.A

    feed_data = data.X

    latent = network.to_z_latent(feed_data)

    latent = sc.AnnData(X=latent)
    latent.obs[cell_type_key] = data.obs[cell_type_key].values

    color = [cell_type_key]

    sc.pp.neighbors(train_data)
    sc.tl.umap(train_data)
    sc.pl.umap(train_data, color=color,
               save=f'_{data_name}_train_data.pdf',
               show=False)

    sc.pp.neighbors(latent)
    sc.tl.umap(latent)
    sc.pl.umap(latent, color=color,
               save=f"_{data_name}_latent.pdf",
               show=False)

    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=20, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-a', '--alpha', type=float, default=0.005, required=False,
                                 help='Alpha coeff in loss term')
    arguments_group.add_argument('-n', '--n_epochs', type=int, default=5000, required=False,
                                 help='Maximum Number of epochs for training')
    arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                                 help='Batch Size')
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.2, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning rate of optimizer')
    arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                                 help='Learning rate of optimizer')
    # arguments_group.add_argument('-s', '--subsample', type=int, default=20000, required=False,
    #                              help='Size of subsampling')
    arguments_group.add_argument('-g', '--gpus', type=int, default=1, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-s', '--arch_style', type=int, default=1, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-v', '--verbose', type=int, default=2, required=False,
                                 help='Learning Rate for Optimizer')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    del args['data']
    if args['do_train'] == 1:
        del args['do_train']
        train_network(data_dict=data_dict, **args)
    visualize_trained_network_results(data_dict, z_dim=args['z_dim'], arch_style=args['arch_style'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
