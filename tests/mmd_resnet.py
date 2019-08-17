import argparse
import math
import os
from random import shuffle

import keras
import numpy as np
import scanpy as sc
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Input, Activation, Dense, BatchNormalization, add
from keras.models import Model
from keras.regularizers import l2
from scipy import sparse

import trvae
from .cost_functions import MMD

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")
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

parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')

arguments_group = parser.add_argument_group("Parameters")
arguments_group.add_argument('-d', '--data', type=str, required=True,
                             help='name of dataset you want to train')
arguments_group.add_argument('-z', '--z_dim', type=int, default=100, required=False,
                             help='latent space dimension')
arguments_group.add_argument('-n', '--n_epochs', type=int, default=500, required=False,
                             help='Maximum Number of epochs for training')
arguments_group.add_argument('-c', '--batch_size', type=int, default=1000, required=False,
                             help='Batch Size')
arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                             help='Batch Size')
arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.5, required=False,
                             help='Dropout ratio')

args = vars(parser.parse_args())
data_dict = DATASETS[args['data']]
data_name = data_dict.get('name', None)
cell_type_key = data_dict.get("cell_type", None)
source_key = data_dict.get('source_key')
target_key = data_dict.get('target_key')

train_path = f"../data/{data_name}/train_{data_name}.h5ad"
valid_path = f"../data/{data_name}/valid_{data_name}.h5ad"

data = sc.read(train_path)
validation = sc.read(valid_path)

if sparse.issparse(data.X):
    data.X = data.X.A
if sparse.issparse(validation.X):
    validation.X = validation.X.A

# =============================== data gathering ====================================
spec_cell_types = data_dict.get('spec_cell_types', None)
cell_types = data.obs[cell_type_key].unique().tolist()

for spec_cell_type in spec_cell_types:
    train_real = data.copy()[~((data.obs['condition'] == target_key) & (data.obs[cell_type_key] == spec_cell_type))]
    train_real_stim = train_real[train_real.obs["condition"] == target_key]
    train_real_ctrl = train_real[train_real.obs["condition"] == source_key]
    train_real_stim = train_real_stim.X

    ind_list = [i for i in range(train_real_stim.shape[0])]
    shuffle(ind_list)
    train_real_stim = train_real_stim[ind_list, :]

    gex_size = train_real_stim.shape[1]
    train_real_ctrl = train_real_ctrl.X
    ind_list = [i for i in range(train_real_ctrl.shape[0])]
    shuffle(ind_list)
    train_real_ctrl = train_real_ctrl[ind_list, :]

    eq = min(len(train_real_ctrl), len(train_real_stim))
    stim_ind = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
    ctrl_ind = np.random.choice(range(len(train_real_ctrl)), size=eq, replace=False)
    ##  selecting equal size for both stimulated and control cells
    train_real_ctrl = train_real_ctrl[ctrl_ind, :]
    train_real_stim = train_real_stim[stim_ind, :]

    # =============================== parameters ====================================
    model_to_use = f"../models/MMDResNet/{data_name}/{spec_cell_type}/"
    os.makedirs(model_to_use, exist_ok=True)

    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2

    calibInput = Input(shape=(gex_size,))
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1)
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(gex_size, activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2)
    block1_output = add([block1_w2, calibInput])
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1)
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(gex_size, activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2)
    block2_output = add([block2_w2, block1_output])
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1)
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(gex_size, activation='linear', kernel_regularizer=l2(l2_penalty),
                      kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2)
    block3_output = add([block3_w2, block2_output])

    calibMMDNet = Model(inputs=calibInput, outputs=block3_output)


    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.1
        epochs_drop = 150.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate


    lrate = LearningRateScheduler(step_decay)

    # train MMD net
    optimizer = keras.optimizers.rmsprop(lr=0.001)

    calibMMDNet.compile(optimizer=optimizer,
                        loss=lambda y_true, y_pred: MMD(block3_output, train_real_stim,
                                                        MMDTargetValidation_split=0.1).KerasCost(y_true, y_pred))

    K.get_session().run(tf.global_variables_initializer())

    sourceLabels = np.zeros(train_real_ctrl.shape[0])
    n_epochs = args['n_epochs']
    batch_size = args['batch_size']
    calibMMDNet.fit(train_real_ctrl, sourceLabels, nb_epoch=n_epochs, batch_size=batch_size, validation_split=0.1,
                    verbose=2, callbacks=[lrate, EarlyStopping(monitor='val_loss', patience=50, mode='auto')])

    path_to_save = f"../results/MMDResNet/{data_name}/{spec_cell_type}"
    sc.settings.figdir = os.path.abspath(path_to_save)
    sc.settings.writedir = os.path.abspath(path_to_save)

    CD4T = data.copy()[data.obs[cell_type_key] == spec_cell_type]
    ctrl_CD4T = data.copy()[(data.obs[cell_type_key] == spec_cell_type) & (data.obs['condition'] == source_key)]
    stim_CD4T = data.copy()[(data.obs[cell_type_key] == spec_cell_type) & (data.obs['condition'] == target_key)]
    if sparse.issparse(ctrl_CD4T.X):
        ctrl_CD4T.X = ctrl_CD4T.X.A
        stim_CD4T.X = stim_CD4T.X.A

    if data_name == "pbmc":
        sc.tl.rank_genes_groups(CD4T, groupby="condition", n_genes=100, method="wilcoxon")
        top_100_genes = CD4T.uns["rank_genes_groups"]["names"][target_key].tolist()
        gene_list = top_100_genes[:10]
    else:
        sc.tl.rank_genes_groups(CD4T, groupby="condition", n_genes=100, method="wilcoxon")
        top_50_down_genes = CD4T.uns["rank_genes_groups"]["names"][source_key].tolist()
        top_50_up_genes = CD4T.uns["rank_genes_groups"]["names"][target_key].tolist()
        top_100_genes = top_50_up_genes + top_50_down_genes
        gene_list = top_50_down_genes[:5] + top_50_up_genes[:5]

    pred_stim = calibMMDNet.predict(ctrl_CD4T.X)
    all_Data = sc.AnnData(np.concatenate([ctrl_CD4T.X, stim_CD4T.X, pred_stim]))
    all_Data.obs["condition"] = ["ctrl"] * len(ctrl_CD4T.X) + ["real_stim"] * len(stim_CD4T.X) + \
                                ["pred_stim"] * len(pred_stim)
    all_Data.var_names = CD4T.var_names

    trvae.plotting.reg_var_plot(all_Data,
                                top_100_genes=top_100_genes,
                                gene_list=gene_list,
                                condition_key='condition',
                                axis_keys={"x": 'predicted', 'y': target_key},
                                labels={'x': 'pred stim', 'y': 'real stim'},
                                legend=False,
                                fontsize=20,
                                textsize=14,
                                title=spec_cell_type,
                                path_to_save=os.path.join(path_to_save,
                                                          f'mmd_resnet_reg_var_{data_name}_{spec_cell_type}.pdf'))

    trvae.plotting.reg_var_plot(all_Data,
                                top_100_genes=top_100_genes,
                                gene_list=gene_list,
                                condition_key='condition',
                                axis_keys={"x": 'predicted', 'y': target_key},
                                labels={'x': 'pred stim', 'y': 'real stim'},
                                legend=False,
                                fontsize=20,
                                textsize=14,
                                title=spec_cell_type,
                                path_to_save=os.path.join(path_to_save,
                                                          f'mmd_resnet_reg_var_{data_name}_{spec_cell_type}.pdf'))
