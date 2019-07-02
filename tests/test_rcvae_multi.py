import argparse
import os

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

DATASETS = {
    "HpolySal": {'name': 'Hpoly+Salmonella', 'need_merge': True,
                 "name1": 'hpoly', 'name2': 'salmonella',
                 'source_conditions': ['Control', 'Hpoly.Day10'],
                 'target_conditions': ['Salmonella'],
                 'preturbation': [('Control', 'Hpoly.Day10', 'ctrl_to_hpoly'),
                                  ('Control', 'Salmonella', 'ctrl_to_sal'),
                                  ('ctrl_to_hpoly', 'Salmonella', '(ctrl_to_hpoly)_to_sal'),
                                  ('ctrl_to_sal', 'hpoly', '(ctrl_to_sal)_to_hpoly'),
                                  ('Hpoly.Day10', 'Control', 'hpoly_to_ctrl')],
                 "cell_type": "cell_label", 'spec_cell_types': ['Stem']},

    "Cytof": {'name': 'cytof', 'need_merge': False,
              'source_conditions': ['Basal', 'Bez', 'Das'],
              'target_conditions': ['Bez+Das'],
              'perturbation': [('Basal', 'Bez', 'Basal_to_Bez', 0, 1),
                               ('Basal', 'Das', 'Basal_to_Das', 0, 2),
                               ('Basal', 'Bez+Das', 'Basal_to_Bez+Das', 0, 3),
                               ('Bez', 'Bez+Das', 'Bez_to_Bez+Das', 1, 3),
                               ('Bez', 'Basal', 'Bez_to_Basal', 1, 0),
                               ('Das', 'Bez+Das', 'Das_to_Bez+Das', 2, 3),
                               ('Das', 'Basal', 'Das_to_Basal', 2, 0),
                               ('Basal_to_Bez', 'Bez+Das', '(Basal_to_Bez)_to_Bez+Das', 1, 3),
                               ('Basal_to_Das', 'Bez+Das', '(Basal_to_Das)_to_Bez+Das', 2, 3),
                               ],
              'label_encoder': {'Basal': 0, 'Bez': 1, 'Das': 2, 'Bez+Das': 3},
              'cell_type': 'cell_label'}

}


def merge_data(data_dict):
    data_name1 = data_dict['name1']
    data_name2 = data_dict['name2']
    target_key2 = data_dict.get('target_key2', None)

    train_data1 = sc.read(filename=f"../data/{data_name1}/train_{data_name1}.h5ad")
    valid_data1 = sc.read(filename=f"../data/{data_name1}/valid_{data_name1}.h5ad")

    train_data2 = sc.read(filename=f"../data/{data_name2}/train_{data_name2}.h5ad")
    valid_data2 = sc.read(filename=f"../data/{data_name2}/valid_{data_name2}.h5ad")

    train_data = train_data1.copy()
    valid_data = valid_data1.copy()

    train_target_data2 = train_data2.copy()[train_data2.obs['condition'] == target_key2]
    valid_target_data2 = valid_data2.copy()[valid_data2.obs['condition'] == target_key2]

    train_data = train_data.concatenate(train_target_data2)
    valid_data = valid_data.concatenate(valid_target_data2)

    return train_data, valid_data


def train_network(data_dict=None,
                  z_dim=100,
                  mmd_dimension=256,
                  alpha=0.001,
                  beta=100,
                  kernel='multi-scale-rbf',
                  n_epochs=500,
                  batch_size=512,
                  early_stop_limit=50,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  arch_style=1,
                  ):
    data_name = data_dict['name']
    source_keys = data_dict.get("source_conditions")
    target_keys = data_dict.get("target_conditions")
    cell_type_key = data_dict.get("cell_type", None)
    need_merge = data_dict.get('need_merge', False)
    label_encoder = data_dict.get('label_encoder', None)

    if need_merge:
        train_data, valid_data = merge_data(data_dict)
    else:
        train_data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
        valid_data = sc.read(f"../data/{data_name}/valid_{data_name}.h5ad")

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if cell_type_key is not None:
        cell_types = train_data.obs[cell_type_key].unique().tolist()
        if spec_cell_type:
            cell_types = spec_cell_type

        for cell_type in cell_types:
            net_train_data = train_data.copy()[~((train_data.obs[cell_type_key] == cell_type) &
                                                 (train_data.obs['condition'].isin(target_keys)))]
            net_valid_data = valid_data.copy()[~((valid_data.obs[cell_type_key] == cell_type) &
                                                 (valid_data.obs['condition'].isin(target_keys)))]

            network = rcvae.RCVAEMulti(x_dimension=net_train_data.shape[1],
                                       z_dimension=z_dim,
                                       n_conditions=len(source_keys),
                                       mmd_dimension=mmd_dimension,
                                       alpha=alpha,
                                       arch_style=arch_style,
                                       beta=beta,
                                       kernel=kernel,
                                       learning_rate=learning_rate,
                                       train_with_fake_labels=False,
                                       model_path=f"../models/RCVAEMulti/{data_name}/{cell_type}/{z_dim}/",
                                       dropout_rate=dropout_rate)

            network.train(net_train_data,
                          label_encoder,
                          use_validation=True,
                          valid_data=net_valid_data,
                          n_epochs=n_epochs,
                          batch_size=batch_size,
                          verbose=2,
                          early_stop_limit=early_stop_limit,
                          shuffle=True,
                          save=True)

            print(f"Model for {cell_type} has been trained")


def visualize_trained_network_results(data_dict, z_dim=100, arch_style=1):
    plt.close("all")
    data_name = data_dict['name']
    source_keys = data_dict.get("source_conditions")
    target_keys = data_dict.get("target_conditions")
    cell_type_key = data_dict.get("cell_type", None)
    need_merge = data_dict.get('need_merge', False)
    label_encoder = data_dict.get('label_encoder', None)

    if need_merge:
        data, _ = merge_data(data_dict)
    else:
        data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")

    cell_types = data.obs[cell_type_key].unique().tolist()

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type:
        cell_types = spec_cell_type

    for cell_type in cell_types:
        path_to_save = f"../results/RCVAEMulti/{data_name}/{cell_type}/{z_dim}/Visualizations/"
        os.makedirs(path_to_save, exist_ok=True)
        sc.settings.figdir = os.path.abspath(path_to_save)

        train_data = data.copy()[
            ~((data.obs['condition'].isin(target_keys)) & (data.obs[cell_type_key] == cell_type))]

        cell_type_adata = data[data.obs[cell_type_key] == cell_type]
        network = rcvae.RCVAEMulti(x_dimension=data.shape[1],
                                   z_dimension=z_dim,
                                   n_conditions=3,
                                   arch_style=arch_style,
                                   model_path=f"../models/RCVAEMulti/{data_name}/{cell_type}/{z_dim}/", )

        network.restore_model()

        if sparse.issparse(data.X):
            data.X = data.X.A

        feed_data = data.X

        train_labels, _ = rcvae.label_encoder(data, label_encoder)
        fake_labels = []

        n_conditions = len(source_keys) + len(target_keys)
        for i in range(n_conditions):
            fake_labels.append(np.zeros(train_labels.shape) + i)

        latent_with_true_labels = network.to_latent(feed_data, train_labels)
        latent_with_fake_labels = [network.to_latent(feed_data, fake_labels[i]) for i in
                                   range(n_conditions)]
        mmd_latent_with_true_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=0)
        mmd_latent_with_fake_labels = [network.to_mmd_layer(network, feed_data, train_labels, feed_fake=i) for i in
                                       range(n_conditions)]

        if data_name in ["pbmc", 'cytof']:
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=10, method="wilcoxon")
            top_10_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_keys[-1]].tolist()
            gene_list = top_10_genes[:10]
        else:
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=10, method="wilcoxon")
            top_5_down_genes = cell_type_adata.uns["rank_genes_groups"]["names"][source_keys[0]].tolist()
            top_5_up_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_keys[-1]].tolist()
            top_10_genes = top_5_up_genes + top_5_down_genes
            gene_list = top_5_down_genes[:5] + top_5_up_genes[:5]
        perturbation_list = data_dict.get("perturbation", [])
        for source, dest, name, source_label, target_label in perturbation_list:
            print(source, dest, name)
            cell_type_adata = visualize_multi_perturbation_between(network, cell_type_adata,
                                                                   source_condition=source, target_condition=dest,
                                                                   name=name,
                                                                   source_label=source_label, target_label=target_label,
                                                                   cell_type=cell_type, data_name=data_name,
                                                                   top_100_genes=top_10_genes, gene_list=gene_list,
                                                                   path_to_save=path_to_save)

        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        if data_name == "cytof":
            color = ['condition']
        else:
            color = ['condition', cell_type_key]

        latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
        latent_with_true_labels.obs['condition'] = data.obs['condition'].values
        latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        latent_with_fake_labels = [sc.AnnData(X=latent_with_fake_labels[i]) for i in range(n_conditions)]
        for i in range(n_conditions):
            latent_with_fake_labels[i].obs['condition'] = data.obs['condition'].values
            latent_with_fake_labels[i].obs[cell_type_key] = data.obs[cell_type_key].values

            sc.pp.neighbors(latent_with_fake_labels[i])
            sc.tl.umap(latent_with_fake_labels[i])
            sc.pl.umap(latent_with_fake_labels[i], color=color,
                       save=f"_{data_name}_{cell_type}_latent_with_fake_labels_{i}",
                       show=False,
                       wspace=0.15,
                       frameon=False)

        mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
        mmd_latent_with_true_labels.obs['condition'] = data.obs['condition'].values
        mmd_latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        mmd_latent_with_fake_labels = [sc.AnnData(X=mmd_latent_with_fake_labels[i]) for i in range(n_conditions)]
        for i in range(n_conditions):
            mmd_latent_with_fake_labels[i].obs['condition'] = data.obs['condition'].values
            mmd_latent_with_fake_labels[i].obs[cell_type_key] = data.obs[cell_type_key].values

            sc.pp.neighbors(mmd_latent_with_fake_labels[i])
            sc.tl.umap(mmd_latent_with_fake_labels[i])
            sc.pl.umap(mmd_latent_with_fake_labels[i], color=color,
                       save=f"_{data_name}_latent_with_fake_labels_{i}",
                       show=False,
                       wspace=0.15,
                       frameon=False)

        sc.pp.neighbors(train_data)
        sc.tl.umap(train_data)
        sc.pl.umap(train_data, color=color,
                   save=f'_{data_name}_{cell_type}_train_data',
                   show=False,
                   wspace=0.15,
                   frameon=False)

        sc.pp.neighbors(latent_with_true_labels)
        sc.tl.umap(latent_with_true_labels)
        sc.pl.umap(latent_with_true_labels, color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_true_labels",
                   show=False,
                   wspace=0.15,
                   frameon=False)

        sc.pp.neighbors(mmd_latent_with_true_labels)
        sc.tl.umap(mmd_latent_with_true_labels)
        sc.pl.umap(mmd_latent_with_true_labels, color=color,
                   save=f"_{data_name}_{cell_type}_mmd_latent_with_true_labels",
                   show=False,
                   wspace=0.15,
                   frameon=False)

        # sc.pl.violin(cell_type_adata, keys=top_100_genes[0], groupby='condition',
        #              save=f"_{data_name}_{cell_type}_{top_100_genes[0]}",
        #              show=False,
        #              wspace=0.15,
        #              frameon=False)

        plt.close("all")


def visualize_multi_perturbation_between(network, adata,
                                         source_condition, target_condition, source_label, target_label, name,
                                         cell_type='', data_name="", top_100_genes=None, gene_list=None,
                                         path_to_save='./'):
    adata_source = adata.copy()[adata.obs['condition'] == source_condition]

    source_labels = np.zeros(adata_source.shape[0]) + source_label
    target_labels = np.zeros(adata_source.shape[0]) + target_label

    pred_target = network.predict(adata_source,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.obs['condition'] = [name] * pred_target.shape[0]
    pred_adata.var_names = adata.var_names

    adata = adata.concatenate(adata, pred_adata)
    print(adata.obs['condition'].unique().tolist())
    rcvae.plotting.reg_mean_plot(adata,
                                 top_100_genes=top_100_genes,
                                 gene_list=gene_list + ['p4EBP1'] + ['pSTAT5'],
                                 condition_key='condition',
                                 axis_keys={"x": f'{name}', 'y': target_condition},
                                 labels={'x': f'{source_condition} to {target_condition}',
                                         'y': f'real {target_condition}'},
                                 legend=False,
                                 fontsize=20,
                                 textsize=14,
                                 path_to_save=os.path.join(path_to_save,
                                                           f'rcvae_reg_mean_{data_name}_{source_condition} to {target_condition}.pdf'))

    rcvae.plotting.reg_var_plot(adata,
                                top_100_genes=top_100_genes,
                                gene_list=gene_list + ['p4EBP1'] + ['pSTAT5'],
                                condition_key='condition',
                                axis_keys={"x": f'{name}', 'y': target_condition},
                                labels={'x': f'{source_condition} to {target_condition}',
                                        'y': f'real {target_condition}'},
                                legend=False,
                                fontsize=20,
                                textsize=14,
                                path_to_save=os.path.join(path_to_save,
                                                          f'rcvae_reg_var_{data_name}_{source_condition} to {target_condition}.pdf'))

    adata_scatter = adata.copy()[adata.obs['condition'].isin([name, target_condition, 'Bez', 'Das', 'Bez+Das'])]
    sc.pl.scatter(adata_scatter, x='p4EBP1', y='pSTAT5', color="condition",
                  save=f'_rcvae_{data_name}_{source_condition} to {target_condition}.png',
                  )

    return adata


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
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.2, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning rate of optimizer')
    arguments_group.add_argument('-y', '--early_stop_limit', type=int, default=50, required=False,
                                 help='do train the network')
    arguments_group.add_argument('-s', '--arch_style', type=int, default=1, required=False,
                                 help='Architecture Style')
    arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                                 help='Learning rate of optimizer')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    del args['data']
    if args['do_train'] == 1:
        del args['do_train']
        train_network(data_dict=data_dict, **args)
    visualize_trained_network_results(data_dict=data_dict, z_dim=args['z_dim'], arch_style=args['arch_style'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
