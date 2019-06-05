import argparse
import os

import numpy as np
import scanpy as sc
from scipy import sparse

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

DATASETS = {
    "atac": {"name": 'atac',
             "matrix_file": "atac_matrix.binary.qc_filtered.mtx.gz",
             "metadata": "cell_metadata.txt",
             'cell_type': 'cell_label',
             'spec_cell_types': [],
             },

}


def train_network(data_dict=None,
                  z_dim=100,
                  mmd_dimension=256,
                  alpha=0.001,
                  beta=100,
                  kernel='multi-scale-rbf',
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  ):
    data_name = data_dict['name']
    metadata_path = data_dict['metadata']
    cell_type_key = data_dict['cell_type']

    train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type is not []:
        cell_types = spec_cell_type

    net_train_data = train_data.copy()

    network = rcvae.VAE(x_dimension=net_train_data.shape[1],
                        z_dimension=z_dim,
                        alpha=alpha,
                        learning_rate=learning_rate,
                        model_path=f"../models/VAE/{data_name}/{z_dim}/",
                        dropout_rate=dropout_rate)

    network.train(net_train_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print(f"Model for {data_name} has been trained")


def visualize_trained_network_results(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict['name']
    metadata_path = data_dict['metadata']
    cell_type_key = data_dict['cell_type']
    spec_cell_type = data_dict.get("spec_cell_types", None)

    data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
    cell_types = data.obs[cell_type_key].unique().tolist()

    path_to_save = f"../results/VAE/{data_name}/{z_dim}/Visualizations/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    train_data = data.copy()

    network = rcvae.VAE(x_dimension=data.shape[1],
                        z_dimension=z_dim,
                        model_path=f"../models/VAE/{data_name}/{z_dim}/", )

    network.restore_model()

    if sparse.issparse(data.X):
        data.X = data.X.A

    feed_data = data.X

    latent = network.to_latent(feed_data)

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


def visualize_trained_network_results_multimodal(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)

    data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    path_to_save = f"../results/RCVAE/{data_name}/{z_dim}/{source_key} to {target_key}/Visualizations/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    network = rcvae.RCVAE(x_dimension=data.shape[1],
                          z_dimension=z_dim,
                          model_path=f"../models/RCVAE/{data_name}/{z_dim}/", )
    network.restore_model()
    if sparse.issparse(data.X):
        data.X = data.X.A

    feed_data = data.X
    train_labels, _ = rcvae.label_encoder(data)
    fake_labels = np.ones(train_labels.shape)
    latent_with_true_labels = network.to_latent(feed_data, train_labels)
    latent_with_fake_labels = network.to_latent(feed_data, fake_labels)
    mmd_latent_with_true_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=False)
    mmd_latent_with_fake_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=True)

    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs['condition'] = data.obs['condition'].values
    # latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
    latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
    # latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs['condition'] = data.obs['condition'].values
    # mmd_latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
    mmd_latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
    # mmd_latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    color = ['condition']

    sc.pp.neighbors(data)
    sc.tl.umap(data)
    sc.pl.umap(data, color=color,
               save=f'_{data_name}_train_data',
               show=False)

    sc.pp.neighbors(latent_with_true_labels)
    sc.tl.umap(latent_with_true_labels)
    sc.pl.umap(latent_with_true_labels, color=color,
               save=f"_{data_name}_latent_with_true_labels",
               show=False)

    sc.pp.neighbors(latent_with_fake_labels)
    sc.tl.umap(latent_with_fake_labels)
    sc.pl.umap(latent_with_fake_labels, color=color,
               save=f"_{data_name}__latent_with_fake_labels",
               show=False)

    sc.pp.neighbors(mmd_latent_with_true_labels)
    sc.tl.umap(mmd_latent_with_true_labels)
    sc.pl.umap(mmd_latent_with_true_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_true_labels",
               show=False)

    sc.pp.neighbors(mmd_latent_with_fake_labels)
    sc.tl.umap(mmd_latent_with_fake_labels)
    sc.pl.umap(mmd_latent_with_fake_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_fake_labels",
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

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    del args['data']
    if args['do_train'] == 1:
        del args['do_train']
        train_network(data_dict=data_dict, **args)
        visualize_trained_network_results_multimodal(data_dict, z_dim=args['z_dim'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
