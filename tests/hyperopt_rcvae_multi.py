from __future__ import print_function

import anndata
import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from scipy import stats

import rcvae

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
                               ('Basal_to_Bez', 'Das', '(Basal_to_Bez)_to_Das', 1, 2),
                               ('Basal_to_Das', 'Bez', '(Basal_to_Das)_to_Bez', 2, 1),
                               ('Basal_to_Bez', 'Bez+Das', '(Basal_to_Bez)_to_Bez+Das', 1, 3),
                               ],
              'label_encoder': {'Basal': 0, 'Bez': 1, 'Das': 2, 'Bez+Das': 3},
              'cell_type': 'cell_label'}

}


def data():
    data_name = 'cytof'
    train_data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    return train_data


def create_model(train_data):
    data_dict = DATASETS['Cytof']
    data_name = data_dict['name']
    target_keys = 'Bez+Das'
    label_encoder = data_dict['label_encoder']

    net_train_data = train_data.copy()[~(train_data.obs['condition'].isin(target_keys))]

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
                               dropout_rate={{choice([10, 20, 50, 75, 100])}})

    network.train(net_train_data,
                  label_encoder,
                  use_validation=False,
                  n_epochs=5000,
                  batch_size={{choice([16, 32, 64, 128, 256, 512, 1024, 2048])}},
                  verbose=2,
                  early_stop_limit=50,
                  shuffle=True,
                  save=False)

    source_adata = train_data.copy()[train_data.obs['condition'] == 'Basal_to_Bez']

    source_labels = np.zeros(source_adata.shape[0]) + 1
    target_labels = np.zeros(source_adata.shape[0]) + 3

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels)

    pred_adata = anndata.AnnData(X=pred_target)
    pred_adata.obs['condition'] = ['(Basal_to_Bez)_to_Bez+Das'] * pred_target.shape[0]
    pred_adata.var_names = source_adata.var_names

    ctrl = pred_adata
    stim = train_data.copy()[train_data.obs['condition'] == 'Bez+Das']

    x = np.var(ctrl.X, axis=0)
    y = np.var(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)

    best_reg_var = r_value
    print('Best Reg Var of model:', best_reg_var)
    return {'loss': -best_reg_var, 'status': STATUS_OK, 'model': network.cvae_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
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
