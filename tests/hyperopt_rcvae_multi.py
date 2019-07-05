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
from scipy import stats, sparse

import rcvae


def data():
    DATASETS = {
        "HpolySal": {'name': 'Hpoly+Salmonella', 'need_merge': True,
                     "name1": 'hpoly', 'name2': 'salmonella',
                     'source_conditions': ['Control', 'Hpoly.Day10'],
                     'target_conditions': ['Salmonella'],
                     'transition': ('ctrl_to_hpoly', 'Salmonella', '(ctrl_to_hpoly)_to_sal'),
                     "cell_type": "cell_label", 'spec_cell_types': ['Stem']},

        "Cytof": {'name': 'cytof', 'need_merge': False,
                  'source_conditions': ['Basal', 'Bez', 'Das', 'Tof'],
                  'target_conditions': ['Bez+Das', 'Bez+Tof'],
                  'transition': ('Bez', 'Bez+Tof', 'Bez_to_Bez+Tof', 1, 5),
                  'label_encoder': {'Basal': 0, 'Bez': 1, 'Das': 2, 'Tof': 3, 'Bez+Das': 4, 'Bez+Tof': 5},
                  'condition': 'condition',
                  'cell_type': 'cell_label'},

        "EndoNorm": {'name': 'endo_norm', 'need_merge': False,
                     'source_conditions': ['Ctrl', 'GLP1', 'Estrogen', 'PEG-insulin', 'Vehicle-STZ', ],
                     'target_conditions': ['GLP1-E', 'GLP1-E + PEG-insulin'],
                     'transition': ('Estrogen', 'GLP1-E', 'Estrogen_to_GLP1-E', 2, 5),
                     'label_encoder': {'Ctrl': 0, 'GLP1': 1, 'Estrogen': 2, 'PEG-insulin': 3, 'Vehicle-STZ': 4,
                                       'GLP1-E': 5,
                                       'GLP1-E + PEG-insulin': 6},
                     'spec_cell_types': ['beta'],
                     'condition': 'treatment',
                     'cell_type': 'groups_named_broad'},

    }
    data_key = "EndoNorm"
    data_dict = DATASETS[data_key]
    data_name = data_dict['name']
    condition_key = data_dict['condition']
    cell_type_key = data_dict['cell_type']
    cell_type = data_dict['spec_cell_types'][0]
    target_keys = data_dict['target_conditions']
    label_encoder = data_dict['label_encoder']

    train_data = sc.read(f"./data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"./data/{data_name}/valid_{data_name}.h5ad")

    net_train_data = train_data.copy()[~(train_data.obs[condition_key].isin(target_keys))]
    net_valid_data = valid_data.copy()[~(valid_data.obs[condition_key].isin(target_keys))]

    n_conditions = len(net_train_data.obs[condition_key].unique().tolist())

    arch_style = 2 if data_name == 'cytof' else 1

    source_condition, target_condition, _, source_label, target_label = data_dict['transition']

    return train_data, valid_data, net_train_data, net_valid_data, condition_key, cell_type_key, cell_type, n_conditions, label_encoder, arch_style, data_name, source_condition, target_condition, source_label, target_label


def create_model(train_data, valid_data,
                 net_train_data, net_valid_data,
                 condition_key, cell_type_key,
                 cell_type, n_conditions,
                 label_encoder,
                 arch_style, data_name,
                 source_condition, target_condition, source_label, target_label):
    if n_conditions == 4:
        z_dim_choices = {{choice([20, 40, 50, 60, 80, 100])}}
        mmd_dim_choices = {{choice([64, 128, 256])}}
        alpha_choices = {{choice([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
        beta_choices = {{choice([50, 100, 200, 400, 600, 800, 1000])}}
        dropout_rate_choices = {{choice([0.1, 0.2, 0.5, 0.75])}}
        batch_size_choices = {{choice([128, 256, 512, 1024, 2048])}}
    else:
        z_dim_choices = {{choice([2, 4, 6, 8, 10, 12])}}
        mmd_dim_choices = {{choice([4, 8, 10, 12, 14, 16])}}
        alpha_choices = {{choice([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
        beta_choices = {{choice([50, 100, 200, 400, 600, 800, 1000])}}
        dropout_rate_choices = {{choice([0.1, 0.2, 0.5, 0.75])}}
        batch_size_choices = {{choice([128, 256, 512, 1024, 2048])}}

    network = rcvae.RCVAEMulti(x_dimension=net_train_data.shape[1],
                               z_dimension=z_dim_choices,
                               arch_style=arch_style,
                               n_conditions=n_conditions,
                               mmd_dimension=mmd_dim_choices,
                               alpha=alpha_choices,
                               beta=beta_choices,
                               kernel='rbf',
                               learning_rate=0.001,
                               model_path=f"../models/RCVAEMulti/{data_name}/hyperopt/",
                               dropout_rate=dropout_rate_choices
                               )

    network.train(net_train_data,
                  label_encoder,
                  condition_key,
                  use_validation=True,
                  valid_data=net_valid_data,
                  n_epochs=10000,
                  batch_size=batch_size_choices,
                  verbose=2,
                  early_stop_limit=50,
                  monitor='val_loss',
                  shuffle=True,
                  save=False)

    cell_type_adata = train_data.copy()[train_data.obs[cell_type_key] == cell_type]
    sc.tl.rank_genes_groups(cell_type_adata, groupby=condition_key, n_genes=100, method="wilcoxon")
    top_100_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_condition].tolist()

    source_adata = cell_type_adata.copy()[cell_type_adata.obs[condition_key] == source_condition]

    source_labels = np.zeros(source_adata.shape[0]) + source_label
    target_labels = np.zeros(source_adata.shape[0]) + target_label

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.var_names = source_adata.var_names

    pred_target = pred_adata.copy()
    real_target = cell_type_adata.copy()[cell_type_adata.obs[condition_key] == target_condition]

    pred_target = pred_target.copy()[:, top_100_genes]
    real_target = real_target.copy()[:, top_100_genes]

    if sparse.issparse(pred_target.X):
        pred_target.X = pred_target.X.A

    if sparse.issparse(real_target.X):
        real_target.X = real_target.X.A

    x_var = np.var(pred_target.X, axis=0)
    y_var = np.var(real_target.X, axis=0)
    m, b, r_value_var, p_value, std_err = stats.linregress(x_var, y_var)

    x_mean = np.mean(pred_target.X, axis=0)
    y_mean = np.mean(real_target.X, axis=0)
    m, b, r_value_mean, p_value, std_err = stats.linregress(x_mean, y_mean)

    best_reg = r_value_var + r_value_mean
    print(f'Best Reg of model: ({r_value_mean}, {r_value_var}, {best_reg})')
    return {'loss': -best_reg, 'status': STATUS_OK, 'model': network}


def visualize_multi_perturbation_between(network, adata, pred_adatas,
                                         source_condition, target_condition, source_label, target_label, name,
                                         cell_type='', data_name="", top_100_genes=None, gene_list=None,
                                         path_to_save='./',
                                         condition_key='condition'):
    adata_source = adata.copy()[adata.obs[condition_key] == source_condition]
    adata_target = adata.copy()[adata.obs[condition_key] == target_condition]

    if adata_source.shape[0] == 0:
        adata_source = pred_adatas.copy()[pred_adatas.obs[condition_key] == source_condition]

    if adata_target.shape[0] == 0:
        adata_target = pred_adatas.copy()[pred_adatas.obs[condition_key] == target_condition]

    source_labels = np.zeros(adata_source.shape[0]) + source_label
    target_labels = np.zeros(adata_source.shape[0]) + target_label

    pred_target = network.predict(adata_source,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.obs[condition_key] = [name] * pred_target.shape[0]
    pred_adata.var_names = adata.var_names

    if sparse.issparse(adata_source.X):
        adata_source.X = adata_source.X.A

    if sparse.issparse(adata_target.X):
        adata_target.X = adata_target.X.A

    if sparse.issparse(pred_adata.X):
        pred_adata.X = pred_adata.X.A

    adata_to_plot = pred_adata.concatenate(adata_target)

    rcvae.plotting.reg_mean_plot(adata_to_plot,
                                 top_100_genes=top_100_genes,
                                 gene_list=gene_list,
                                 condition_key=condition_key,
                                 axis_keys={"x": f'{name}', 'y': target_condition},
                                 labels={'x': f'{source_condition} to {target_condition}',
                                         'y': f'real {target_condition}'},
                                 legend=False,
                                 fontsize=20,
                                 textsize=14,
                                 path_to_save=os.path.join(path_to_save,
                                                           f'rcvae_reg_mean_{data_name}_{source_condition} to {target_condition}.pdf'))

    rcvae.plotting.reg_var_plot(adata_to_plot,
                                top_100_genes=top_100_genes,
                                gene_list=gene_list,
                                condition_key=condition_key,
                                axis_keys={"x": f'{name}', 'y': target_condition},
                                labels={'x': f'{source_condition} to {target_condition}',
                                        'y': f'real {target_condition}'},
                                legend=False,
                                fontsize=20,
                                textsize=14,
                                path_to_save=os.path.join(path_to_save,
                                                          f'rcvae_reg_var_{data_name}_{source_condition} to {target_condition}.pdf'))

    return pred_adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-n', '--max_evals', type=int, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())
    data_key = args['data']

    best_run, best_network = optim.minimize(model=create_model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=args['max_evals'],
                                            trials=Trials())
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)
    # print(best_model)
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
                     'condition': 'condition',
                     "cell_type": "cell_label", 'spec_cell_types': ['Stem']},

        "Cytof": {'name': 'cytof', 'need_merge': False,
                  'source_conditions': ['Basal', 'Bez', 'Das', 'Tof'],
                  'target_conditions': ['Bez+Das', 'Bez+Tof'],
                  'perturbation': [('Basal', 'Bez', 'Basal_to_Bez', 0, 1),
                                   ('Basal', 'Das', 'Basal_to_Das', 0, 2),
                                   ('Basal', 'Tof', 'Basal_to_Das', 0, 3),
                                   ('Basal', 'Bez+Das', 'Basal_to_Bez+Das', 0, 4),
                                   ('Basal', 'Bez+Tof', 'Basal_to_Bez+Tof', 0, 5),
                                   ('Bez', 'Bez+Das', 'Bez_to_Bez+Das', 1, 4),
                                   ('Bez', 'Bez+Tof', 'Bez_to_Bez+Tof', 1, 5),
                                   ('Bez', 'Basal', 'Bez_to_Basal', 1, 0),
                                   ('Das', 'Bez+Das', 'Das_to_Bez+Das', 2, 4),
                                   ('Das', 'Basal', 'Das_to_Basal', 2, 0),
                                   ('Basal_to_Bez', 'Bez+Das', '(Basal_to_Bez)_to_Bez+Das', 1, 4),
                                   ('Basal_to_Bez', 'Bez+Tof', '(Basal_to_Bez)_to_Bez+Tof', 1, 5),
                                   ('Basal_to_Das', 'Bez+Das', '(Basal_to_Das)_to_Bez+Das', 2, 4),
                                   ],
                  'label_encoder': {'Basal': 0, 'Bez': 1, 'Das': 2, 'Tof': 3, 'Bez+Das': 4, 'Bez+Tof': 5},
                  'spec_cell_types': ['None'],
                  'condition': 'condition',
                  'cell_type': 'cell_label'},

        "EndoNorm": {'name': 'endo_norm', 'need_merge': False,
                     'source_conditions': ['Ctrl', 'GLP1', 'Estrogen', 'PEG-insulin', 'Vehicle-STZ', ],
                     'target_conditions': ['GLP1-E', 'GLP1-E + PEG-insulin'],
                     'perturbation': [('Ctrl', 'GLP1', 'Ctrl_to_GLP1', 0, 1),
                                      ('Ctrl', 'Estrogen', 'Ctrl_to_Estrogen', 0, 2),
                                      ('Ctrl', 'PEG-insulin', 'Ctrl_to_PEG-insulin', 0, 3),
                                      ('GLP1', 'GLP1-E', 'GLP1_to_GLP1-E', 1, 5),
                                      ('GLP1', 'GLP1-E + PEG-insulin', 'GLP1_to_GLP1-E + PEG-insulin', 1, 6),
                                      ('Estrogen', 'GLP1-E', 'Estrogen_to_GLP1-E', 2, 5),
                                      ('Estrogen', 'GLP1-E + PEG-insulin', 'Estrogen_to_GLP1-E + PEG-insulin', 2, 6),
                                      ('PEG-insulin', 'GLP1-E + PEG-insulin', 'PEG-insulin_to_GLP1-E + PEG-insulin', 3,
                                       6),
                                      ('Estrogen_to_GLP1-E', 'GLP1-E + PEG-insulin',
                                       '(Estrogen_to_GLP1-E)_to_GLP1-E + PEG-insulin', 5, 6),
                                      ('GLP1_to_GLP1-E', 'GLP1-E + PEG-insulin',
                                       '(GLP1_to_GLP1-E)_to_GLP1-E + PEG-insulin',
                                       5, 6),
                                      ],
                     'label_encoder': {'Ctrl': 0, 'GLP1': 1, 'Estrogen': 2, 'PEG-insulin': 3, 'Vehicle-STZ': 4,
                                       'GLP1-E': 5,
                                       'GLP1-E + PEG-insulin': 6},
                     'spec_cell_types': ['beta'],
                     'condition': 'treatment',
                     'cell_type': 'groups_named_broad'},

    }

    data_dict = DATASETS[data_key]

    data_name = data_dict['name']
    condition_key = data_dict['condition']
    cell_type_key = data_dict['cell_type']
    source_keys = data_dict['source_conditions']
    target_keys = data_dict['target_conditions']
    label_encoder = data_dict['label_encoder']
    cell_type = data_dict['spec_cell_types'][0]

    train_data = sc.read(f"./data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"./data/{data_name}/valid_{data_name}.h5ad")

    net_train_data = train_data.copy()[~(train_data.obs[condition_key].isin(target_keys))]
    net_valid_data = valid_data.copy()[~(valid_data.obs[condition_key].isin(target_keys))]

    arch_style = 2 if data_name == 'cytof' else 1

    path_to_save = f"./results/RCVAEMulti/hyperopt/{data_name}/{cell_type}/{best_network.z_dim}/Visualizations/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    n_conditions = len(net_train_data.obs[condition_key].unique().tolist())

    train_labels, _ = rcvae.label_encoder(train_data, label_encoder, condition_key)
    fake_labels = []
    for i in range(n_conditions):
        fake_labels.append(np.zeros(train_labels.shape) + i)

    feed_data = train_data.X

    latent_with_true_labels = best_network.to_latent(feed_data, train_labels)
    latent_with_fake_labels = [best_network.to_latent(feed_data, fake_labels[i]) for i in
                               range(n_conditions)]
    mmd_latent_with_true_labels = best_network.to_mmd_layer(best_network, feed_data, train_labels, feed_fake=0)
    mmd_latent_with_fake_labels = [best_network.to_mmd_layer(best_network, feed_data, train_labels, feed_fake=i) for i
                                   in
                                   range(n_conditions)]

    cell_type_adata = train_data[train_data.obs[cell_type_key] == cell_type]

    if data_name in ["pbmc", 'endo_norm']:
        sc.tl.rank_genes_groups(cell_type_adata, groupby=condition_key, n_genes=100, method="wilcoxon")
        top_100_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_keys[-1]].tolist()
        gene_list = top_100_genes[:10]
    elif data_name in ['cytof']:
        sc.tl.rank_genes_groups(cell_type_adata, groupby=condition_key, n_genes=10, method="wilcoxon")
        top_100_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_keys[-1]].tolist()
        gene_list = top_100_genes[:10]
    else:
        sc.tl.rank_genes_groups(cell_type_adata, groupby=condition_key, n_genes=10, method="wilcoxon")
        top_50_down_genes = cell_type_adata.uns["rank_genes_groups"]["names"][source_keys[0]].tolist()
        top_50_up_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_keys[-1]].tolist()
        top_100_genes = top_50_up_genes + top_50_down_genes
        gene_list = top_50_down_genes[:5] + top_50_up_genes[:5]
    perturbation_list = data_dict.get("perturbation", [])
    pred_adatas = None
    for source, dest, name, source_label, target_label in perturbation_list:
        print(source, dest, name)
        pred_adata = visualize_multi_perturbation_between(best_network, cell_type_adata, pred_adatas,
                                                          source_condition=source, target_condition=dest,
                                                          name=name,
                                                          source_label=source_label, target_label=target_label,
                                                          cell_type=cell_type, data_name=data_name,
                                                          top_100_genes=top_100_genes, gene_list=gene_list,
                                                          path_to_save=path_to_save,
                                                          condition_key=condition_key)
        if pred_adatas is None:
            pred_adatas = pred_adata
        else:
            pred_adatas = pred_adatas.concatenate(pred_adata)

    pred_adatas.write_h5ad(filename=f"../data/reconstructed/RCVAEMulti/{data_name}.h5ad")

    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

    if data_name == "cytof":
        color = [condition_key]
    else:
        color = [condition_key, cell_type_key]

    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs[condition_key] = train_data.obs[condition_key].values
    latent_with_true_labels.obs[cell_type_key] = train_data.obs[cell_type_key].values

    latent_with_fake_labels = [sc.AnnData(X=latent_with_fake_labels[i]) for i in range(n_conditions)]
    for i in range(n_conditions):
        latent_with_fake_labels[i].obs[condition_key] = train_data.obs[condition_key].values
        latent_with_fake_labels[i].obs[cell_type_key] = train_data.obs[cell_type_key].values

        sc.pp.neighbors(latent_with_fake_labels[i])
        sc.tl.umap(latent_with_fake_labels[i])
        sc.pl.umap(latent_with_fake_labels[i], color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_fake_labels_{i}",
                   show=False,
                   wspace=0.15,
                   frameon=False)

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs[condition_key] = train_data.obs[condition_key].values
    mmd_latent_with_true_labels.obs[cell_type_key] = train_data.obs[cell_type_key].values

    mmd_latent_with_fake_labels = [sc.AnnData(X=mmd_latent_with_fake_labels[i]) for i in range(n_conditions)]
    for i in range(n_conditions):
        mmd_latent_with_fake_labels[i].obs[condition_key] = train_data.obs[condition_key].values
        mmd_latent_with_fake_labels[i].obs[cell_type_key] = train_data.obs[cell_type_key].values

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

    for target_condition in target_keys:
        pred_adata = pred_adatas[pred_adatas.obs[condition_key].str.endswith(target_condition)]
        violin_adata = cell_type_adata.concatenate(pred_adata)
        for gene in top_100_genes[:3]:
            sc.pl.violin(violin_adata, keys=gene, groupby=condition_key,
                         save=f"_{data_name}_{cell_type}_{gene}_{target_condition}.pdf",
                         show=False,
                         wspace=0.2,
                         rotation=90,
                         frameon=False)

    plt.close("all")

    print("All Done!")
    print(best_run)

"""
    best run for MMD-CVAE:
    alpha = xxx, 
    beta = xx,
    kernel = rbf,
    n_epochs = 5000,
    z_dim = xxx
"""
