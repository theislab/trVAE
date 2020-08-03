import argparse
import csv
import os

import scanpy as sc
from keras import backend as K

import trvae
from trvae.utils import normalize_hvg, train_test_split

DATASETS = {
    "Kang": {'name': 'kang',
             'source_conditions': ['control'],
             'target_conditions': ['stimulated'],
             "cell_type_key": "cell_type", "condition_key": "condition",
             'spec_cell_types': ['NK'],
             "label_encoder": {"control": 0, "stimulated": 1}},
    "Haber": {'name': 'haber',
              'source_conditions': ['Control'],
              'target_conditions': ['Hpoly.Day10'],
              "cell_type_key": "cell_label", "condition_key": "condition",
              'spec_cell_types': ['Tuft'],
              "label_encoder": {"Control": 0, "Hpoly.Day10": 1}},

}


def create_data(data_dict):
    data_name = data_dict['name']
    source_keys = data_dict.get("source_conditions")
    target_keys = data_dict.get("target_conditions")
    cell_type_key = data_dict.get("cell_type_key", None)
    condition_key = data_dict.get('condition_key', 'condition')
    spec_cell_type = data_dict.get("spec_cell_types", None)[0]

    adata = sc.read(f"./data/{data_name}/{data_name}_normalized.h5ad")
    adata = adata[adata.obs[condition_key].isin(source_keys + target_keys)]

    if adata.shape[1] > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']]

    train_adata, valid_adata = train_test_split(adata, 0.80)

    net_train_adata = train_adata.copy()[~((train_adata.obs[cell_type_key] == spec_cell_type) &
                                           (train_adata.obs[condition_key].isin(target_keys)))]
    net_valid_adata = valid_adata.copy()[~((valid_adata.obs[cell_type_key] == spec_cell_type) &
                                           (valid_adata.obs[condition_key].isin(target_keys)))]
    return adata, net_train_adata, net_valid_adata


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
                  early_stop_limit=50,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  verbose=2,
                  adata=None,
                  net_train_adata=None,
                  net_valid_adata=None,
                  ):
    print(f"Training network with beta = {beta}")
    data_name = data_dict['name']
    cell_type_key = data_dict.get("cell_type_key", None)
    condition_key = data_dict.get('condition_key', 'condition')
    label_encoder = data_dict.get('label_encoder', None)
    spec_cell_type = data_dict.get("spec_cell_types", None)[0]

    n_conditions = len(net_train_adata.obs[condition_key].unique().tolist())

    network = trvae.archs.trVAEMulti(x_dimension=net_train_adata.shape[1],
                                     z_dimension=z_dim,
                                     n_conditions=n_conditions,
                                     mmd_dimension=mmd_dim,
                                     alpha=alpha,
                                     beta=beta,
                                     eta=eta,
                                     kernel=kernel,
                                     learning_rate=learning_rate,
                                     output_activation="relu",
                                     model_path=f"./models/trVAEMulti/Monitor/{data_name}/{spec_cell_type}/{filename}/{beta}/",
                                     dropout_rate=dropout_rate,
                                     )

    network.train(net_train_adata,
                  net_valid_adata,
                  label_encoder,
                  condition_key,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  early_stop_limit=early_stop_limit,
                  lr_reducer=int(0.8 * early_stop_limit),
                  verbose=verbose,
                  save=True,
                  monitor_best=False)

    # Calculate ASW for MMD Layer

    encoder_labels, _ = trvae.utils.label_encoder(adata, label_encoder, condition_key)
    mmd_latent = network.to_mmd_layer(adata, encoder_labels, feed_fake=-1, return_adata=True)

    asw = trvae.mt.asw(mmd_latent, condition_key)
    ebm = trvae.mt.entropy_batch_mixing(mmd_latent, condition_key, n_pools=1)
    ari = trvae.mt.ari(mmd_latent, cell_type_key)
    nmi = trvae.mt.nmi(mmd_latent, cell_type_key)

    _, rec, mmd = network.get_reconstruction_error(net_valid_adata, condition_key)

    row = [alpha, eta, z_dim, mmd_dim, beta, asw, nmi, ari, ebm, rec, mmd]
    with open(f"./{filename}.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    file.close()

    os.makedirs(f"./results/Monitor/{filename}/", exist_ok=True)
    sc.settings.figdir = f"./results/Monitor/{filename}/"

    sc.pp.neighbors(mmd_latent)
    sc.tl.umap(mmd_latent)
    sc.pl.umap(mmd_latent, color=condition_key, frameon=False, title="", save=f"_trVAE_MMD_condition_{beta}.pdf")
    sc.pl.umap(mmd_latent, color=cell_type_key, frameon=False, title="", save=f"_trVAE_MMD_cell_type_{beta}.pdf")

    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, required=True,
                                 help='z_dim')
    arguments_group.add_argument('-m', '--mmd_dim', type=int, required=True,
                                 help='mmd_dim')
    arguments_group.add_argument('-e', '--eta', type=float, required=True,
                                 help='eta')
    arguments_group.add_argument('-b', '--batch_size', type=float, required=False, default=512,
                                 help='batch_size')
    arguments_group.add_argument('-l', '--early_stop_limit', type=int, required=False, default=50,
                                 help='patience')

    args = vars(parser.parse_args())
    row = ["Alpha", "Eta", "Z", "MMD", "beta", "ASW", "NMI", "ARI", "EBM", "sse_loss", 'mmd_loss']

    data_dict = DATASETS[args['data']]
    prev_batch_size = args['batch_size']
    del args['data']
    adata, net_train_adata, net_valid_adata = create_data(data_dict)
    for alpha in [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
        filename = f"alpha={alpha}, eta={args['eta']}, Z={int(args['z_dim'])}, MMD={int(args['mmd_dim'])}"
        with open(f"./{filename}.csv", 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        file.close()
        for beta in [1e5, 5e4, 1e4, 5000, 1000, 500, 250, 200, 100, 50, 10, 5, 1, 0]:
            if beta == 0:
                args['batch_size'] = 64
            else:
                args['batch_size'] = prev_batch_size
            if beta <= 500:
                args['early_stop_limit'] = 500
            train_network(data_dict=data_dict, alpha=alpha, beta=beta, filename=filename,
                          adata=adata, net_train_adata=net_train_adata, net_valid_adata=net_valid_adata,
                          **args)
