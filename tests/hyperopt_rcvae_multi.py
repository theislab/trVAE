from __future__ import print_function

import argparse

import anndata
import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from scipy import stats

import rcvae


def data(data_key):
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
                  'transition': ('Basal_to_Bez', 'Bez+Tof', '(Basal_to_Bez)_to_Bez+Tof', 1, 5),
                  'label_encoder': {'Basal': 0, 'Bez': 1, 'Das': 2, 'Tof': 3, 'Bez+Das': 4, 'Bez+Tof': 5},
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

    data_dict = DATASETS[data_key]
    data_name = data_dict['name']
    condition_key = data_dict['condition']
    target_keys = data_dict['target_conditions']
    label_encoder = data_dict['label_encoder']

    train_data = sc.read(f"./data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"./data/{data_name}/valid_{data_name}.h5ad")

    net_train_data = train_data.copy()[~(train_data.obs[condition_key].isin(target_keys))]
    net_valid_data = valid_data.copy()[~(valid_data.obs[condition_key].isin(target_keys))]

    n_conditions = len(net_train_data.obs[condition_key].unique().tolist())

    arch_style = 2 if data_name == 'cytof' else 1

    def inner_data():
        return train_data, valid_data, net_train_data, net_valid_data, condition_key, n_conditions, label_encoder, arch_style, data_name

    return inner_data


def create_model(train_data, valid_data, net_train_data, net_valid_data, condition_key, n_conditions, label_encoder, arch_style, data_name):
    network = rcvae.RCVAEMulti(x_dimension=net_train_data.shape[1],
                               z_dimension={{choice([20, 40, 50, 60, 80, 100, 200])}},
                               arch_style=arch_style,
                               n_conditions=n_conditions,
                               mmd_dimension={{choice([4, 6, 8, 10, 12, 14, 16])}},
                               alpha={{choice([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}},
                               beta={{choice([50, 100, 200, 400, 600, 800, 1000])}},
                               kernel='rbf',
                               learning_rate={{choice([0.001, 0.0001, 0.00001])}},
                               model_path=f"../models/RCVAEMulti/{data_name}/hyperopt/",
                               dropout_rate={{choice([0.1, 0.2, 0.5, 0.75])}})

    network.train(net_train_data,
                  label_encoder,
                  condition_key,
                  use_validation=True,
                  valid_data=net_valid_data,
                  n_epochs=5000,
                  batch_size={{choice([32, 64, 128, 256, 512, 1024, 2048])}},
                  verbose=2,
                  early_stop_limit=50,
                  monitor='val_loss',
                  shuffle=True,
                  save=False)

    source_condition, target_condition, _, source_label, target_label = data_dict['transition']
    source_adata = train_data.copy()[train_data.obs[condition_key] == source_condition]

    source_labels = np.zeros(source_adata.shape[0]) + source_label
    target_labels = np.zeros(source_adata.shape[0]) + target_label

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.var_names = source_adata.var_names

    pred_target = pred_adata.copy()
    real_target = train_data.copy()[train_data.obs[condition_key] == target_condition]

    x_var = np.var(pred_target.X, axis=0)
    y_var = np.var(real_target.X, axis=0)
    m, b, r_value_var, p_value, std_err = stats.linregress(x_var, y_var)

    x_mean = np.mean(pred_target.X, axis=0)
    y_mean = np.mean(real_target.X, axis=0)
    m, b, r_value_mean, p_value, std_err = stats.linregress(x_mean, y_mean)

    best_reg = r_value_var + r_value_mean
    print(f'Best Reg of model: ({r_value_mean}, {r_value_var}, {best_reg})')
    return {'loss': -best_reg, 'status': STATUS_OK, 'model': network.cvae_model}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')

    args = vars(parser.parse_args())
    data_key = args['data']

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data(data_key),
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=Trials())
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)

"""
    best run for MMD-CVAE:
    alpha = xxx, 
    beta = xx,
    kernel = rbf,
    n_epochs = 5000,
    z_dim = xxx
"""
