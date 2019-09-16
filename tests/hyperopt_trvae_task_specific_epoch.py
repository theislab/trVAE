from __future__ import print_function

import argparse

import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from scipy import stats

import trvae
from trvae.utils import train_test_split, remove_sparsity


def data():
    DATASETS = {
        "Haber": {'name': 'haber', 'need_merge': False,
                  'source_conditions': ['Control'],
                  'target_conditions': ['Hpoly.Day10', 'Hpoly.Day3', 'Salmonella'],
                  'transition': ('Control', 'Hpoly.Day10', 'Control_to_Hpoly.Day10'),
                  'condition_encoder': {'Control': 0, 'Hpoly.Day3': 1, 'Hpoly.Day10': 2, 'Salmonella': 3},
                  'condition_key': 'condition',
                  'cell_type_key': 'cell_label'},
        "Kang": {'name': 'kang', 'need_merge': False,
                 'source_conditions': ['control'],
                 'target_conditions': ['stimulated'],
                 'transition': ('control', 'stimulated', 'control_to_stimulated'),
                 'condition_encoder': {'control': 0, 'stimulated': 1},
                 'condition_key': 'condition',
                 'cell_type_key': 'cell_type'},
    }

    data_key = "Kang"
    cell_type = ["NK"]

    data_dict = DATASETS[data_key]

    data_name = data_dict['name']
    condition_key = data_dict['condition_key']
    cell_type_key = data_dict['cell_type_key']
    target_conditions = data_dict['target_conditions']
    condition_encoder = data_dict['condition_encoder']

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")

    if adata.shape[1] > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']]

    train_adata, valid_adata = train_test_split(adata, 0.80)

    if cell_type and target_conditions:
        net_train_adata = train_adata.copy()[~((train_adata.obs[cell_type_key].isin(cell_type)) &
                                               (train_adata.obs[condition_key].isin(target_conditions)))]
        net_valid_adata = valid_adata.copy()[~((valid_adata.obs[cell_type_key].isin(cell_type)) &
                                               (valid_adata.obs[condition_key].isin(target_conditions)))]
    elif target_conditions:
        net_train_adata = train_adata.copy()[~(train_adata.obs[condition_key].isin(target_conditions))]
        net_valid_adata = valid_adata.copy()[~(valid_adata.obs[condition_key].isin(target_conditions))]

    else:
        net_train_adata = train_adata.copy()
        net_valid_adata = valid_adata.copy()

    source_condition, target_condition, _ = data_dict['transition']

    return train_adata, net_train_adata, net_valid_adata, condition_key, cell_type_key, cell_type[0], condition_encoder, data_name, source_condition, target_condition


def create_model(train_adata,
                 net_train_adata, net_valid_adata,
                 condition_key, cell_type_key,
                 cell_type, condition_encoder,
                 data_name, source_condition, target_condition):
    n_conditions = len(net_train_adata.obs[condition_key].unique().tolist())

    z_dim_choices = {{choice([10, 20, 40, 50, 60, 80, 100])}}
    mmd_dim_choices = {{choice([64, 128, 256])}}

    alpha_choices = {{choice([0.001, 0.0001, 0.00001, 0.000001])}}
    beta_choices = {{choice([1, 5, 10, 20, 40, 50, 100])}}
    eta_choices = {{choice([1, 2, 5, 10, 50, 100])}}
    n_epochs_cvae_choices = {{choice([1, 2, 5, 10])}}
    n_epochs_mmd_choices = {{choice([1, 2, 5, 10])}}
    batch_size_choices = {{choice([128, 256, 512, 1024, 1500])}}
    dropout_rate_choices = {{choice([0.1, 0.2, 0.5])}}

    network = trvae.archs.trVAETaskSpecific(x_dimension=net_train_adata.shape[1],
                                            z_dimension=z_dim_choices,
                                            n_conditions=n_conditions,
                                            mmd_dimension=mmd_dim_choices,
                                            alpha=alpha_choices,
                                            beta=beta_choices,
                                            eta=eta_choices,
                                            kernel='multi-scale-rbf',
                                            learning_rate=0.001,
                                            clip_value=1e6,
                                            loss_fn='mse',
                                            model_path=f"./models/trVAETaskSpecific/hyperopt/{data_name}/{cell_type}/{target_condition}/",
                                            dropout_rate=dropout_rate_choices,
                                            output_activation="relu",
                                            )

    network.train_epoch(net_train_adata,
                        net_valid_adata,
                        condition_encoder,
                        condition_key,
                        n_epochs=10000,
                        n_epochs_cvae=n_epochs_cvae_choices,
                        n_epochs_mmd=n_epochs_mmd_choices,
                        batch_size=batch_size_choices,
                        verbose=2,
                        early_stop_limit=100,
                        lr_reducer=80,
                        monitor='val_loss',
                        shuffle=True,
                        save=False)

    cell_type_adata = train_adata.copy()[train_adata.obs[cell_type_key] == cell_type]

    sc.tl.rank_genes_groups(cell_type_adata,
                            key_added='up_reg_genes',
                            groupby=condition_key,
                            groups=[target_condition],
                            reference=source_condition,
                            n_genes=10)

    sc.tl.rank_genes_groups(cell_type_adata,
                            key_added='down_reg_genes',
                            groupby=condition_key,
                            groups=[source_condition],
                            reference=target_condition,
                            n_genes=10)

    up_genes = cell_type_adata.uns['up_reg_genes']['names'][target_condition].tolist()
    down_genes = cell_type_adata.uns['down_reg_genes']['names'][source_condition].tolist()

    top_genes = up_genes + down_genes

    source_adata = cell_type_adata.copy()[cell_type_adata.obs[condition_key] == source_condition]

    source_label = condition_encoder[source_condition]
    target_label = condition_encoder[target_condition]

    source_labels = np.zeros(source_adata.shape[0]) + source_label
    target_labels = np.zeros(source_adata.shape[0]) + target_label

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    real_target = cell_type_adata.copy()[cell_type_adata.obs[condition_key] == target_condition]

    real_target = remove_sparsity(real_target)

    pred_target = pred_target[:, top_genes]
    real_target = real_target[:, top_genes]

    x_var = np.var(pred_target.X, axis=0)
    y_var = np.var(real_target.X, axis=0)
    m, b, r_value_var, p_value, std_err = stats.linregress(x_var, y_var)
    r_value_var = r_value_var ** 2

    x_mean = np.mean(pred_target.X, axis=0)
    y_mean = np.mean(real_target.X, axis=0)
    m, b, r_value_mean, p_value, std_err = stats.linregress(x_mean, y_mean)
    r_value_mean = r_value_mean ** 2

    best_mean_diff = np.abs(np.mean(x_mean - y_mean))
    best_var_diff = np.abs(np.var(x_var - y_var))
    objective = r_value_mean + r_value_var
    print(f'Reg_mean_diff: {r_value_mean}, Reg_var_all: {r_value_var})')
    print(f'Mean diff: {best_mean_diff}, Var_diff: {best_var_diff}')
    print(
        f'alpha = {network.alpha}, beta = {network.beta}, eta={network.eta}, z_dim = {network.z_dim}, mmd_dim = {network.mmd_dim}, batch_size = {batch_size_choices}, dropout_rate = {network.dr_rate}, lr = {network.lr}')
    return {'loss': -objective, 'status': STATUS_OK}


def predict_between_conditions(network, adata, pred_adatas, source_condition, source_label, target_label, name,
                               condition_key='condition'):
    adata_source = adata.copy()[adata.obs[condition_key] == source_condition]

    if adata_source.shape[0] == 0:
        adata_source = pred_adatas.copy()[pred_adatas.obs[condition_key] == source_condition]

    source_labels = np.zeros(adata_source.shape[0]) + source_label
    target_labels = np.zeros(adata_source.shape[0]) + target_label

    pred_adata = network.predict(adata_source,
                                 encoder_labels=source_labels,
                                 decoder_labels=target_labels)
    pred_adata.obs[condition_key] = name
    pred_adata = remove_sparsity(pred_adata)

    return pred_adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-n', '--max_evals', type=int, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())
    data_key = "Kang"
    cell_type = ["NK"]

    best_run, best_network = optim.minimize(model=create_model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=args['max_evals'],
                                            trials=Trials())
    # DATASETS = {
    #     "Haber": {'name': 'haber', 'need_merge': False,
    #               'source_conditions': ['Control'],
    #               'target_conditions': ['Hpoly.Day10', 'Hpoly.Day3', 'Salmonella'],
    #               'transition': [
    #                   ('Control', 'Hpoly.Day3', 'Control_to_Hpoly.Day3'),
    #                   ('Control', 'Hpoly.Day10', 'Control_to_Hpoly.Day10'),
    #                   ('Control', 'Salmonella', 'Control_to_Salmonella'),
    #                   ('Hpoly.Day3', 'Hpoly.Day10', 'Hpoly.Day3_to_Hpoly.Day10'),
    #                   ('Control_to_Hpoly.Day3', 'Hpoly.Day10', 'Control_to_Hpoly.Day3_to_Hpoly.Day10'),
    #               ],
    #               'condition_encoder': {'Control': 0, 'Hpoly.Day3': 1, 'Hpoly.Day10': 2, 'Salmonella': 3},
    #               'conditions': ['Control', 'Hpoly.Day3', 'Hpoly.Day10', 'Salmonella'],
    #               'condition': 'condition',
    #               'cell_type': 'cell_label'},
    # }
    # data_dict = DATASETS[data_key]
    # data_name = data_dict['name']
    # condition_key = data_dict['condition']
    # cell_type_key = data_dict['cell_type']
    # source_keys = data_dict['source_conditions']
    # target_keys = data_dict['target_conditions']
    # label_encoder = data_dict['condition_encoder']
    # conditions = data_dict.get('conditions', None)
    #
    # data = sc.read(f"./data/{data_name}/{data_name}.h5ad")
    # if conditions:
    #     data = data[data.obs[condition_key].isin(conditions)]
    # train_adata, valid_adata = train_test_split(data, 0.80)
    #
    # if cell_type:
    #     cell_type = cell_type[0]
    #
    # n_conditions = len(train_adata.obs[condition_key].unique().tolist())
    #
    # train_labels, _ = trvae.utils.label_encoder(train_adata, label_encoder, condition_key)
    # cell_type_adata = train_adata[train_adata.obs[cell_type_key] == cell_type]
    #
    # perturbation_list = data_dict.get("transition", [])
    # pred_adatas = None
    # for source, target, name, in perturbation_list:
    #     print(source, target, name)
    #     if len(source.split("_to_")) > 1:
    #         source = source.split("_to_")[-1]
    #     source_label = label_encoder[source]
    #     target_label = label_encoder[target]
    #     pred_adata = predict_between_conditions(best_network, cell_type_adata, pred_adatas,
    #                                             source_condition=source, source_label=source_label,
    #                                             target_label=target_label, name=name,
    #                                             condition_key=condition_key)
    #     if pred_adatas is None:
    #         pred_adatas = pred_adata
    #     else:
    #         pred_adatas = pred_adatas.concatenate(pred_adata)
    # path_to_save = f"./data/reconstructed/trVAEMulti/hyperopt/{data_name}/{cell_type}/"
    # os.makedirs(path_to_save, exist_ok=True)
    # pred_adatas.write_h5ad(filename=os.path.join(path_to_save, f"reconstructed.h5ad"))

    # best_network.save_model()
    print("All Done!")
    print(best_run)
