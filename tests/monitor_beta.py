import argparse
import csv
import os

import numpy as np
import scanpy as sc

import trvae
from trvae.utils import normalize, train_test_split

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

DATASETS = {
    "Kang": {'name': 'kang',
             'source_conditions': ['CTRL'],
             'target_conditions': ['STIM'],
             "cell_type_key": "cell_type", "condition_key": "condition",
             'spec_cell_types': ['NK'],
             "label_encoder": {"CTRL": 0, "STIM": 1}},

}


def train_network(data_dict=None,
                  filename=None,
                  z_dim=20,
                  mmd_dim=128,
                  alpha=0.00005,
                  beta=100,
                  eta=1.0,
                  kernel='multi-scale-rbf',
                  n_epochs=5000,
                  batch_size=512,
                  early_stop_limit=20,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  verbose=2,
                  ):
    print(f"Training network with beta = {beta}")
    data_name = data_dict['name']
    target_keys = data_dict.get("target_conditions")
    cell_type_key = data_dict.get("cell_type_key", None)
    label_encoder = data_dict.get('label_encoder', None)
    condition_key = data_dict.get('condition_key', 'condition')

    adata = sc.read(f"../data/{data_name}/{data_name}_normalized.h5ad")

    if adata.shape[0] > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']]

    train_data, valid_data = train_test_split(adata, 0.80)

    spec_cell_type = data_dict.get("spec_cell_types", None)[0]

    net_train_data = train_data.copy()[~((train_data.obs[cell_type_key] == spec_cell_type) &
                                         (train_data.obs[condition_key].isin(target_keys)))]
    net_valid_data = valid_data.copy()[~((valid_data.obs[cell_type_key] == spec_cell_type) &
                                         (valid_data.obs[condition_key].isin(target_keys)))]
    n_conditions = len(net_train_data.obs[condition_key].unique().tolist())

    network = trvae.archs.trVAEMulti(x_dimension=net_train_data.shape[1],
                                     z_dimension=z_dim,
                                     n_conditions=n_conditions,
                                     mmd_dimension=mmd_dim,
                                     alpha=alpha,
                                     beta=beta,
                                     eta=eta,
                                     kernel=kernel,
                                     learning_rate=learning_rate,
                                     output_activation="relu",
                                     model_path=f"../models/trVAEMulti/Monitor/{data_name}/{spec_cell_type}/{filename}/{beta}/",
                                     dropout_rate=dropout_rate,
                                     )

    network.train(net_train_data,
                  net_valid_data,
                  label_encoder,
                  condition_key,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  early_stop_limit=early_stop_limit,
                  lr_reducer=int(0.8 * early_stop_limit),
                  verbose=verbose,
                  save=True)

    # Calculate ASW for MMD Layer

    encoder_labels, _ = trvae.utils.label_encoder(adata, label_encoder, condition_key)
    mmd_latent = network.to_mmd_layer(adata, encoder_labels, feed_fake=-1, return_adata=True)

    asw = trvae.mt.asw(mmd_latent, condition_key)
    ebm = trvae.mt.entropy_batch_mixing(mmd_latent, condition_key, n_pools=1)
    ari = trvae.mt.ari(mmd_latent, cell_type_key)
    nmi = trvae.mt.nmi(mmd_latent, cell_type_key)

    _, rec, mmd = network.get_reconstruction_error(net_valid_data, condition_key)

    row = [alpha, eta, z_dim, mmd_dim, beta, asw, nmi, ari, ebm, rec, mmd]
    with open(f"../{filename}.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, required=True,
                                 help='z_dim')
    arguments_group.add_argument('-m', '--mmd_dim', type=int, required=True,
                                 help='mmd_dim')
    arguments_group.add_argument('-a', '--alpha', type=float, required=True,
                                 help='alpha')
    arguments_group.add_argument('-e', '--eta', type=float, required=True,
                                 help='eta')

    args = vars(parser.parse_args())
    row = ["Alpha", "Eta", "Z", "MMD", "beta", "ASW", "NMI", "ARI", "EBM", "sse_loss", 'mmd_loss']
    filename = f"alpha={args['alpha']}, eta={args['eta']}, Z={int(args['z_dim'])}, MMD={int(args['mmd_dim'])}"
    with open(f"../{filename}.csv", 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    file.close()

    data_dict = DATASETS[args['data']]
    del args['data']
    for beta in np.arange(0, 1000, 50).tolist():
        train_network(data_dict=data_dict, beta=beta, filename=filename, **args)
