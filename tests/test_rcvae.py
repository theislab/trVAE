import argparse
import os

import numpy as np
import scanpy as sc

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

DATASETS = {
    "Pancreas": {"name": 'pancreas', "source_key": "Baron", "target_key": "Segerstolpe",
                 'train_celltypes': ['alpha', 'beta', 'ductal', 'acinar', 'delta', 'gamma'],
                 'test_celltypes': ['beta']},
    "Kang": {"name": 'mnist', "source_key": 1, "target_key": 7, "resize": 28, 'size': 28, "n_channels": 1},
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
                  ):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    if data_name == "pancreas":
        train_celltypes = data_dict.get("train_celltypes", None)
        test_celltypes = data_dict.get("test_celltypes", None)

        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        train_data = train_data[train_data.obs['sample'].isin([source_key, target_key])]
        train_data.obs['condition'] = train_data.obs['sample']
        train_data = train_data[train_data.obs['celltype'].isin(train_celltypes)]
        train_data = train_data[
            ~((train_data.obs["condition"] == target_key) & (train_data.obs['celltype'].isin(test_celltypes)))]

        source_data = train_data.copy()[train_data.obs["condition"] == source_key].X
        target_data = train_data.copy()[train_data.obs["condition"] == target_key].X

    source_labels = np.zeros(shape=source_data.shape[0])
    target_labels = np.ones(shape=target_data.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)
    train_data.obs["condition"] = train_labels

    network = rcvae.RCVAE(x_dimension=source_data.shape[1],
                          z_dimension=z_dim,
                          mmd_dimension=mmd_dimension,
                          alpha=alpha,
                          beta=beta,
                          kernel=kernel,
                          train_with_fake_labels=True,
                          model_path=f"../models/{data_name}/{z_dim}/",
                          dropout_rate=dropout_rate)

    print(source_data.shape, target_data.shape)

    network.train(train_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


def visualize_trained_network_results(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)

    path_to_save = f"../results/{data_name}/{z_dim}/{source_key} to {target_key}/UMAPs/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    if data_name == 'pancreas':
        train_celltypes = data_dict.get("train_celltypes", None)
        test_celltypes = data_dict.get("test_celltypes", None)

        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        train_data = train_data[train_data.obs['sample'].isin([source_key, target_key])]
        train_data.obs['condition'] = train_data.obs['sample']
        train_data = train_data[train_data.obs['celltype'].isin(train_celltypes)]
        test_cell_types_data = train_data[(train_data.obs['celltype'].isin(test_celltypes))]
        train_data = train_data[
            ~((train_data.obs["condition"] == target_key) & (train_data.obs['celltype'].isin(test_celltypes)))]

        source_data = train_data.copy()[train_data.obs["condition"] == source_key].X
        target_data = train_data.copy()[train_data.obs["condition"] == target_key].X

    source_labels = np.zeros(shape=source_data.shape[0])
    target_labels = np.ones(shape=target_data.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)
    fake_labels = np.ones(train_labels.shape)
    train_data.obs['condition'] = train_labels

    network = rcvae.RCVAE(x_dimension=train_data.shape[1],
                          z_dimension=z_dim,
                          model_path=f"../models/{data_name}/{z_dim}/", )

    network.restore_model()

    train_data_feed = train_data.X

    latent_with_true_labels = network.to_latent(train_data_feed, train_labels)
    latent_with_fake_labels = network.to_latent(train_data_feed, fake_labels)
    mmd_latent_with_true_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=False)
    mmd_latent_with_fake_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=True)
    test_cell_types_data_feed = test_cell_types_data.copy()[test_cell_types_data.obs['condition'] == source_key]
    pred_celltypes = network.predict(test_cell_types_data_feed,
                                     encoder_labels=np.zeros((test_cell_types_data_feed.shape[0], 1)),
                                     decoder_labels=np.zeros((test_cell_types_data_feed.shape[0], 1)))


    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs['condition'] = train_data.obs['condition'].values

    latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
    latent_with_fake_labels.obs['condition'] = train_data.obs['condition'].values

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs['condition'] = train_data.obs['condition'].values

    mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
    mmd_latent_with_fake_labels.obs['condition'] = train_data.obs['condition'].values

    if data_name.__contains__("mnist"):
        latent_with_true_labels.obs['labels'] = train_data.obs['labels']
        latent_with_fake_labels.obs['labels'] = train_data.obs['labels']
        mmd_latent_with_true_labels.obs['labels'] = train_data.obs['labels']
        mmd_latent_with_fake_labels.obs['labels'] = train_data.obs['labels']

        color = ['condition', 'labels']
    else:
        color = ['condition']

    sc.pp.neighbors(train_data)
    sc.tl.umap(train_data)
    sc.pl.umap(train_data, color=color,
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
               save=f"_{data_name}_latent_with_fake_labels",
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
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.4, required=False,
                                 help='Dropout ratio')
    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    del args['data']
    train_network(data_dict=data_dict, **args)
    visualize_trained_network_results(data_dict,
                                      z_dim=args['z_dim'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
