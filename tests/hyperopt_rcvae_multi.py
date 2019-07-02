from __future__ import print_function

import anndata
import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from scipy import stats

import rcvae


def data():
    data_name = 'cytof'
    train_data = sc.read(f"./data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"./data/{data_name}/valid_{data_name}.h5ad")
    return train_data, valid_data


def create_model(train_data, valid_data):
    data_name = 'cytof'
    target_keys = ['Bez+Das', 'Bez+Tof']
    label_encoder = {'Basal': 0, 'Bez': 1, 'Das': 2, 'Tof': 3, 'Bez+Das': 4, 'Bez+Tof': 5}

    net_train_data = train_data.copy()[~(train_data.obs['condition'].isin(target_keys))]
    net_valid_data = valid_data.copy()[~(valid_data.obs['condition'].isin(target_keys))]

    network = rcvae.RCVAEMulti(x_dimension=net_train_data.shape[1],
                               z_dimension={{choice([2, 4, 6, 8, 10, 12])}},
                               arch_style=2,
                               n_conditions=3,
                               mmd_dimension={{choice([4, 6, 8, 10, 12, 14, 16])}},
                               alpha={{choice([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}},
                               beta={{choice([50, 100, 200, 400, 600, 800, 1000])}},
                               kernel='rbf',
                               learning_rate={{choice([0.01, 0.001, 0.0001, 0.00001])}},
                               model_path=f"../models/RCVAEMulti/{data_name}/hyperopt/",
                               dropout_rate={{choice([0.1, 0.2, 0.5, 0.75])}})

    network.train(net_train_data,
                  label_encoder,
                  use_validation=True,
                  valid_data=net_valid_data,
                  n_epochs=5000,
                  batch_size={{choice([16, 32, 64, 128, 256, 512, 1024, 2048])}},
                  verbose=2,
                  early_stop_limit=50,
                  monitor='val_loss',
                  shuffle=True,
                  save=False)

    source_adata = train_data.copy()[train_data.obs['condition'] == 'Bez']

    source_labels = np.zeros(source_adata.shape[0]) + 1
    target_labels = np.zeros(source_adata.shape[0]) + 3

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.obs['condition'] = ['Bez_to_Bez+Das'] * pred_target.shape[0]
    pred_adata.var_names = source_adata.var_names

    ctrl = pred_adata
    stim = train_data.copy()[train_data.obs['condition'] == 'Bez+Das']

    x_var = np.var(ctrl.X, axis=0)
    y_var = np.var(stim.X, axis=0)
    m, b, r_value_var, p_value, std_err = stats.linregress(x_var, y_var)

    x_mean = np.mean(ctrl.X, axis=0)
    y_mean = np.mean(stim.X, axis=0)
    m, b, r_value_mean, p_value, std_err = stats.linregress(x_mean, y_mean)

    best_reg = 1.5 * r_value_var + r_value_mean
    print('Best Reg Var of model:', best_reg)
    return {'loss': -best_reg, 'status': STATUS_OK, 'model': network.cvae_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
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
