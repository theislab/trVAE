from __future__ import print_function

import argparse

import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras.utils import to_categorical

import trvae
from trvae.utils import train_test_split, remove_sparsity, label_encoder


def data():
    DATASETS = {
        "ATAC": {'name': 'multimodal', 'need_merge': False,
                 'domain_encoder': {'RNA-seq': 0, 'ATAC-seq': 1},
                 'source_domains': ['RNA-seq'],
                 'target_domains': ['ATAC-seq'],
                 'domain': 'modality',
                 'label': 'cell_subclass'},
        "PBMC_ATAC": {'name': 'pbmc_atac', 'need_merge': False,
                      'domain_encoder': {'RNA-seq': 0, 'ATAC-seq': 1},
                      'source_domains': ['RNA-seq'],
                      'target_domains': ['ATAC-seq'],
                      'domain': 'domain',
                      'label': 'cell_type'},
        "Pancreas": {'name': 'pancreas', 'need_merge': False,
                     'domain_encoder': {'Baron': 0, 'Muraro': 1, 'Wang': 2, 'Segerstolpe': 3},
                     'source_domains': ['Baron'],
                     'target_domains': ['Muraro', 'Wang', 'Segerstolpe'],
                     'cell_types': ['acinar', 'beta', 'delta', 'ductal', 'gamma'],
                     'domain': 'sample',
                     'label': 'celltype'},
    }
    data_key = "Pancreas"
    data_dict = DATASETS[data_key]
    data_name = data_dict['name']
    domain_key = data_dict['domain']
    label_key = data_dict['label']
    source_domains = data_dict['source_domains']
    target_domains = data_dict['target_domains']
    domain_encoder = data_dict['domain_encoder']
    celltypes = data_dict.get('cell_types', None)

    adata = sc.read(f"./data/{data_name}/{data_name}.h5ad")

    if celltypes:
        adata = adata.copy()[adata.obs[label_key].isin(celltypes)]

    train_adata, valid_adata = train_test_split(adata, 0.80)

    net_train_adata = train_adata.copy()
    net_valid_adata = valid_adata.copy()

    return net_train_adata, net_valid_adata, domain_key, label_key, source_domains, target_domains, domain_encoder


def create_model(net_train_adata, net_valid_adata,
                 domain_key, label_key,
                 source_domains, target_domains,
                 domain_encoder):
    z_dim_choices = {{choice([20, 40, 50, 60, 80, 100])}}
    mmd_dim_choices = {{choice([64, 128, 256])}}

    alpha_choices = {{choice([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}}
    beta_choices = {{choice([1, 100, 500, 1000, 1500, 2000, 5000])}}
    gamma_choices = {{choice([1, 10, 100, 1000, 5000, 10000])}}
    eta_choices = {{choice([0.01, 0.1, 1, 5, 10, 50])}}
    batch_size_choices = {{choice([128, 256, 512, 1024, 1500])}}
    dropout_rate_choices = {{choice([0.1, 0.2, 0.5])}}

    n_labels = len(net_train_adata.obs[label_key].unique().tolist())
    n_domains = len(net_train_adata.obs[domain_key].unique().tolist())

    network = trvae.archs.trVAEATAC(x_dimension=net_train_adata.shape[1],
                                    z_dimension=z_dim_choices,
                                    mmd_dimension=mmd_dim_choices,
                                    learning_rate=0.001,
                                    alpha=alpha_choices,
                                    beta=beta_choices,
                                    gamma=gamma_choices,
                                    eta=eta_choices,
                                    model_path=f"./models/trVAEATAC/hyperopt/Pancreas/",
                                    n_labels=n_labels,
                                    n_domains=n_domains,
                                    output_activation='leaky_relu',
                                    mmd_computation_way="1",
                                    dropout_rate=dropout_rate_choices
                                    )

    network.train(net_train_adata,
                  net_valid_adata,
                  domain_key,
                  label_key,
                  source_key=source_domains,
                  target_key=target_domains,
                  domain_encoder=domain_encoder,
                  n_epochs=10000,
                  batch_size=batch_size_choices,
                  early_stop_limit=500,
                  lr_reducer=0,
                  )

    target_adata_train = net_train_adata.copy()[net_train_adata.obs[domain_key].isin(target_domains)]
    target_adata_valid = net_valid_adata.copy()[net_valid_adata.obs[domain_key].isin(target_domains)]

    target_adata = target_adata_train.concatenate(target_adata_valid)
    target_adata = remove_sparsity(target_adata)

    target_adata_domains_encoded, _ = label_encoder(target_adata, condition_key=domain_key,
                                                    label_encoder=domain_encoder)
    target_adata_domains_onehot = to_categorical(target_adata_domains_encoded, num_classes=n_domains)

    target_adata_classes_encoded = network.label_enc.transform(target_adata.obs[label_key].values)
    target_adata_classes_onehot = to_categorical(target_adata_classes_encoded, num_classes=n_labels)

    x_target = [target_adata.X,
                target_adata_domains_onehot,
                target_adata_domains_onehot]
    y_target = target_adata_classes_onehot

    _, target_acc = network.classifier_model.evaluate(x_target, y_target, verbose=0)
    objective = -target_acc
    print(
        f'alpha = {network.alpha}, beta = {network.beta}, eta={network.eta}, z_dim = {network.z_dim}, mmd_dim = {network.mmd_dim}, batch_size = {batch_size_choices}, dropout_rate = {network.dr_rate}, gamma = {network.gamma}')
    return {'loss': objective, 'status': STATUS_OK}


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
    print("All Done!")
    print(best_run)
