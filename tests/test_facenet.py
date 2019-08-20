import argparse
import os

import numpy as np
import scanpy as sc
from scipy import sparse

import trvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

DATASETS = {
    "CelebA": {"name": 'celeba', "gender": "Male", 'attribute': "Smiling",
               "width": 64, 'height': 64, "n_channels": 3},

    # "Horse2Zebra": {"name": "h2z", "source_key": "horse", "target_key": "zebra", "size": 256, "n_channels": 3,
    #                 "resize": 64},
    # "Apple2Orange": {"name": "a2o", "source_key": "apple", "target_key": "orange", "size": 256, "n_channels": 3,
    #                  "resize": 64}
}


def train_network(data_dict=None,
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  preprocess=True,
                  learning_rate=0.001,
                  gpus=1,
                  max_size=50000,
                  early_stopping_limit=50,
                  ):
    data_name = data_dict['name']
    img_width = data_dict.get("width", None)
    img_height = data_dict.get("height", None)
    n_channels = data_dict.get("n_channels", None)
    attribute = data_dict.get('attribute', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = trvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             landmark_path="../data/celeba/list_landmarks_align_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=max_size,
                                             img_width=img_width,
                                             img_height=img_height,
                                             restore=True,
                                             save=True)

        if sparse.issparse(data.X):
            data.X = data.X.A

        data.obs.loc[(data.obs['labels'] == -1) & (data.obs['condition'] == -1), 'label'] = 0
        data.obs.loc[(data.obs['labels'] == -1) & (data.obs['condition'] == 1), 'label'] = 1
        data.obs.loc[(data.obs['labels'] == 1) & (data.obs['condition'] == -1), 'label'] = 2
        data.obs.loc[(data.obs['labels'] == 1) & (data.obs['condition'] == 1), 'label'] = 3

        if preprocess:
            data.X /= 255.0
    else:
        data = sc.read(f"../data/{data_name}/{data_name}.h5ad")

        if preprocess:
            data.X /= 255.0

    train_size = int(data.shape[0] * 0.85)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = data[train_idx, :]
    valid_data = data[test_idx, :]

    network = trvae.FaceNet(x_dimension=(img_width, img_height, n_channels),
                            learning_rate=learning_rate,
                            model_path=f"../models/",
                            gpus=gpus,
                            dropout_rate=dropout_rate)

    network.train(train_data,
                  use_validation=True,
                  valid_adata=valid_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=early_stopping_limit,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-n', '--n_epochs', type=int, default=5000, required=False,
                                 help='Maximum Number of epochs for training')
    arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                                 help='Batch Size')
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.4, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-w', '--width', type=int, default=0, required=False,
                                 help='Image Width to be resize')
    arguments_group.add_argument('-e', '--height', type=int, default=0, required=False,
                                 help='Image Height to be resize')
    arguments_group.add_argument('-p', '--preprocess', type=int, default=True, required=False,
                                 help='do preprocess images')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-g', '--gpus', type=int, default=1, required=False,
                                 help='Learning Rate for Optimizer')
    arguments_group.add_argument('-x', '--max_size', type=int, default=50000, required=False,
                                 help='Max Size for CelebA')
    arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                                 help='do train the network')
    arguments_group.add_argument('-y', '--early_stopping_limit', type=int, default=50, required=False,
                                 help='do train the network')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    if args['width'] > 0 and args['height'] > 0:
        data_dict['width'] = args['width']
        data_dict['height'] = args['height']

    if args['preprocess'] == 0:
        args['preprocess'] = False
    else:
        args['preprocess'] = True

    if args['max_size'] == 0:
        args['max_size'] = None

    del args['data']
    del args['width']
    del args['height']
    if args['do_train'] > 0:
        del args['do_train']
        train_network(data_dict=data_dict, **args)
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
