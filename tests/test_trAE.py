import argparse
import os

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

import trvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

DATASETS = {
    "Pancreas": {"name": 'pancreas', "source_key": "Baron", "target_key": "Segerstolpe",
                 'train_celltypes': ['alpha', 'beta', 'ductal', 'acinar', 'delta', 'gamma'],
                 'test_celltypes': ['beta'],
                 'cell_type': 'celltype'},

    "PBMC": {"name": 'pbmc', "source_key": "control", "target_key": 'stimulated',
             "cell_type": "cell_type", 'spec_cell_types': ['CD4T', "CD14+Mono", "FCGR3A+Mono"]},

    "Hpoly": {"name": 'hpoly', "source_key": "Control", "target_key": 'Hpoly.Day10',
              "cell_type": "cell_label", 'spec_cell_types': ['Tuft', 'Endocrine']},

    "Salmonella": {"name": 'salmonella', "source_key": "Control", "target_key": 'Salmonella',
                   "cell_type": "cell_label", 'spec_cell_types': ['Tuft', 'Endocrine']},
}


def train_network(data_dict=None,
                  z_dim=100,
                  beta=0.001,
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  ):
    data_name = data_dict['name']
    target_key = data_dict.get('target_key', None)
    cell_type_key = data_dict.get("cell_type", None)

    train_data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"../data/{data_name}/valid_{data_name}.h5ad")
    cell_types = train_data.obs[cell_type_key].unique().tolist()

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type is not []:
        cell_types = spec_cell_type

    for cell_type in cell_types:
        net_train_data = train_data.copy()[
            ~((train_data.obs[cell_type_key] == cell_type) & (train_data.obs['condition'] == target_key))]
        net_valid_data = valid_data.copy()[
            ~((valid_data.obs[cell_type_key] == cell_type) & (valid_data.obs['condition'] == target_key))]

        network = trvae.trAE(x_dimension=net_train_data.shape[1],
                             z_dimension=z_dim,
                             beta=beta,
                             model_path=f"../models/RAE/{data_name}/{cell_type}/{z_dim}/",
                             dropout_rate=dropout_rate)

        network.train(net_train_data,
                      use_validation=True,
                      valid_adata=net_valid_data,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      early_stop_limit=100,
                      shuffle=True)

        print(f"Model for {cell_type} has been trained")


def visualize_trained_network_results(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    cell_type_key = data_dict.get("cell_type", None)

    data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    cell_types = data.obs[cell_type_key].unique().tolist()

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type is not []:
        cell_types = spec_cell_type

    for cell_type in cell_types:
        path_to_save = f"../results/RAE/{data_name}/{cell_type}/{z_dim}/{source_key} to {target_key}/Visualizations/"
        os.makedirs(path_to_save, exist_ok=True)
        sc.settings.figdir = os.path.abspath(path_to_save)

        train_data = data.copy()[~((data.obs['condition'] == target_key) & (data.obs[cell_type_key] == cell_type))]

        cell_type_adata = data[data.obs[cell_type_key] == cell_type]

        network = trvae.trAE(x_dimension=data.shape[1],
                             z_dimension=z_dim,
                             model_path=f"../models/RAE/{data_name}/{cell_type}/{z_dim}/")

        network.restore_model()

        if sparse.issparse(data.X):
            data.X = data.X.A

        feed_data = data.X

        latent_with_true_labels = network.to_z_latent(feed_data)
        latent_with_fake_labels = network.to_z_latent(feed_data)

        cell_type_ctrl = cell_type_adata.copy()[cell_type_adata.obs['condition'] == source_key]
        print(cell_type_ctrl.shape, cell_type_adata.shape)

        pred_celltypes = network.predict(cell_type_ctrl)
        pred_adata = anndata.AnnData(X=pred_celltypes)
        pred_adata.obs['condition'] = ['predicted'] * pred_adata.shape[0]
        pred_adata.var = cell_type_adata.var

        if data_name == "pbmc":
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=100, method="wilcoxon")
            top_100_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_key].tolist()
            gene_list = top_100_genes[:10]
        else:
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=100, method="wilcoxon")
            top_50_down_genes = cell_type_adata.uns["rank_genes_groups"]["names"][source_key].tolist()
            top_50_up_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_key].tolist()
            top_100_genes = top_50_up_genes + top_50_down_genes
            gene_list = top_50_down_genes[:5] + top_50_up_genes[:5]

        cell_type_adata = cell_type_adata.concatenate(pred_adata)

        trvae.plotting.reg_mean_plot(cell_type_adata,
                                     top_100_genes=top_100_genes,
                                     gene_list=gene_list,
                                     condition_key='condition',
                                     axis_keys={"x": 'predicted', 'y': target_key},
                                     labels={'x': 'pred stim', 'y': 'real stim'},
                                     legend=False,
                                     fontsize=20,
                                     textsize=14,
                                     title=cell_type,
                                     path_to_save=os.path.join(path_to_save,
                                                               f'rcvae_reg_mean_{data_name}_{cell_type}.pdf'))

        trvae.plotting.reg_var_plot(cell_type_adata,
                                    top_100_genes=top_100_genes,
                                    gene_list=gene_list,
                                    condition_key='condition',
                                    axis_keys={"x": 'predicted', 'y': target_key},
                                    labels={'x': 'pred stim', 'y': 'real stim'},
                                    legend=False,
                                    fontsize=20,
                                    textsize=14,
                                    title=cell_type,
                                    path_to_save=os.path.join(path_to_save,
                                                              f'rcvae_reg_var_{data_name}_{cell_type}.pdf'))

        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
        latent_with_true_labels.obs['condition'] = data.obs['condition'].values
        latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
        latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
        latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        color = ['condition', cell_type_key]

        sc.pp.neighbors(train_data)
        sc.tl.umap(train_data)
        sc.pl.umap(train_data, color=color,
                   save=f'_{data_name}_{cell_type}_train_data',
                   show=False)

        sc.pp.neighbors(latent_with_true_labels)
        sc.tl.umap(latent_with_true_labels)
        sc.pl.umap(latent_with_true_labels, color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_true_labels",
                   show=False)

        sc.pp.neighbors(latent_with_fake_labels)
        sc.tl.umap(latent_with_fake_labels)
        sc.pl.umap(latent_with_fake_labels, color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_fake_labels",
                   show=False)

        sc.pl.violin(cell_type_adata, keys=top_100_genes[0], groupby='condition',
                     save=f"_{data_name}_{cell_type}_{top_100_genes[0]}",
                     show=False)

        plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=20, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-b', '--beta', type=float, default=0.005, required=False,
                                 help='Beta coeff in loss term')
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
    visualize_trained_network_results(data_dict, z_dim=args['z_dim'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
