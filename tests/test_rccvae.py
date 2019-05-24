import argparse
import os

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

FASHION_MNIST_CLASS_DICT = {
    0: "T-shirt or top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

DATASETS = {
    "CelebA": {"name": 'celeba', "gender": "Male", 'attribute': "Smiling", 'source_key': -1, "target_key": 1,
               "resize": 64, "n_channels": 3},

    "MNIST": {"name": 'mnist', "source_key": 1, "target_key": 7,
              "train_digits": [], "test_digits": [],
              "resize": 28, 'size': 28, "n_channels": 1},

    "ThinMNIST": {"name": 'thin_mnist', "source_key": "normal", "target_key": "thin",
                  'train_digits': [1, 3, 6, 7], 'test_digits': [0, 2, 4, 5, 8, 9],
                  "resize": 28, 'size': 28,
                  "n_channels": 1},

    "ThickMNIST": {"name": 'thick_mnist', "source_key": "normal", "target_key": "thick",
                   'train_digits': [1, 3, 6, 7], 'test_digits': [0, 2, 4, 5, 8, 9],
                   "resize": 28, 'size': 28,
                   "n_channels": 1},

    "FashionMNIST": {"name": "fashion_mnist", "source_key": FASHION_MNIST_CLASS_DICT[0],
                     "target_key": FASHION_MNIST_CLASS_DICT[1], "resize": 28, 'size': 28, "n_channels": 1},

    "Horse2Zebra": {"name": "h2z", "source_key": "horse", "target_key": "zebra", "size": 256, "n_channels": 3,
                    "resize": 64},
    "Apple2Orange": {"name": "a2o", "source_key": "apple", "target_key": "orange", "size": 256, "n_channels": 3,
                     "resize": 64}
}


def train_network(data_dict=None,
                  z_dim=100,
                  mmd_dimension=256,
                  alpha=0.001,
                  beta=100,
                  kernel='multi-scale-rbf',
                  n_epochs=500,
                  batch_size=512,
                  dropout_rate=0.2,
                  arch_style=1,
                  preprocess=True,
                  learning_rate=0.001
                  ):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_resize = data_dict.get("resize", None)
    n_channels = data_dict.get("n_channels", None)
    train_digits = data_dict.get("train_digits", None)
    test_digits = data_dict.get("test_digits", None)
    attribute = data_dict.get('attribute', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = rcvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=50000,
                                             img_resize=img_resize,
                                             restore=True,
                                             save=True)

        if sparse.issparse(data.X):
            data.X = data.X.A

        source_images = data.copy()[data.obs['condition'] == source_key].X
        target_images = data.copy()[data.obs['condition'] == target_key].X

        source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
        target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))

        if preprocess:
            source_images /= 255.0
            target_images /= 255.0
    else:
        data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        img_size = data_dict.get("size", None)

        source_images = data.copy()[data.obs["condition"] == source_key].X
        target_images = data.copy()[data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_size, img_size, n_channels))
        target_images = np.reshape(target_images, (-1, img_size, img_size, n_channels))
        if img_resize != img_size:
            source_images = rcvae.resize_image(source_images, img_resize)
            target_images = rcvae.resize_image(target_images, img_resize)

            source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
            target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))
        if preprocess:
            source_images /= 255.0
            target_images /= 255.0

    source_labels = np.zeros(shape=source_images.shape[0])
    target_labels = np.ones(shape=target_images.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)

    train_images = np.concatenate([source_images, target_images], axis=0)
    train_images = np.reshape(train_images, (-1, np.prod(source_images.shape[1:])))
    if data_name.__contains__('mnist'):
        preprocessed_data = anndata.AnnData(X=train_images)
        preprocessed_data.obs["condition"] = train_labels
        preprocessed_data.obs['labels'] = data.obs['labels'].values
        data = preprocessed_data.copy()

    train_size = int(data.shape[0] * 0.85)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    data_train = data[train_idx, :]
    data_valid = data[test_idx, :]
    print(data_train.shape, data_valid.shape)

    if train_digits is not None:
        train_data = data_train.copy()[
            ~((data_train.obs['labels'].isin(test_digits)) & (data_train.obs['condition'] == 1))]
        valid_data = data_valid.copy()[
            ~((data_valid.obs['labels'].isin(test_digits)) & (data_valid.obs['condition'] == 1))]
    elif data_name == "celeba":
        train_data = data_train.copy()[
            ~((data_train.obs['labels'] == -1) & (data_train.obs['condition'] == target_key))]
        valid_data = data_valid.copy()[
            ~((data_valid.obs['labels'] == -1) & (data_valid.obs['condition'] == target_key))]
    else:
        train_data = data_train.copy()
        valid_data = data_valid.copy()

    network = rcvae.RCCVAE(x_dimension=source_images.shape[1:],
                           z_dimension=z_dim,
                           mmd_dimension=mmd_dimension,
                           alpha=alpha,
                           beta=beta,
                           kernel=kernel,
                           arch_style=arch_style,
                           train_with_fake_labels=False,
                           learning_rate=learning_rate,
                           model_path=f"../models/{data_name}-{img_resize}-{preprocess}/{arch_style}-{z_dim}/",
                           dropout_rate=dropout_rate)

    print(train_data.shape, valid_data.shape)

    network.train(train_data,
                  use_validation=True,
                  valid_data=valid_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


def evaluate_network(data_dict=None, z_dim=100, n_files=5, k=5, arch_style=1, preprocess=True):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_resize = data_dict.get("resize", None)
    img_size = data_dict.get("size", None)
    n_channels = data_dict.get('n_channels', None)
    train_digits = data_dict.get('train_digits', None)
    test_digits = data_dict.get('test_digits', None)
    attribute = data_dict.get('attribute', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = rcvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=5000,
                                             img_resize=img_resize,
                                             restore=True,
                                             save=False)

        valid_data = data.copy()[data.obs['labels'] == -1]  # get females (Male = -1)
        if sparse.issparse(valid_data.X):
            valid_data.X = valid_data.X.A

        source_images = valid_data[valid_data.obs["condition"] == source_key].X
        target_images = valid_data[valid_data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
        target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))

        if preprocess:
            source_images /= 255.0
            target_images /= 255.0
    else:
        data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        if train_digits is not None:
            valid_data = data[data.obs['labels'].isin(test_digits)]

        source_images = valid_data[valid_data.obs["condition"] == source_key].X
        target_images = valid_data[valid_data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_size, img_size, n_channels))
        target_images = np.reshape(target_images, (-1, img_size, img_size, n_channels))
        if img_resize != img_size:
            source_images = rcvae.resize_image(source_images, img_resize)
            target_images = rcvae.resize_image(target_images, img_resize)

            source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
            target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))
        if preprocess:
            source_images /= 255.0
            target_images /= 255.0

    image_shape = (img_resize, img_resize, n_channels)

    source_labels = np.zeros(shape=source_images.shape[0])
    target_labels = np.ones(shape=target_images.shape[0])

    source_images = np.reshape(source_images, (-1, np.prod(image_shape)))
    target_images = np.reshape(target_images, (-1, np.prod(image_shape)))

    source_data = anndata.AnnData(X=source_images)
    source_data.obs["condition"] = source_labels

    target_data = anndata.AnnData(X=target_images)
    target_data.obs["condition"] = target_labels

    network = rcvae.RCCVAE(x_dimension=image_shape,
                           z_dimension=z_dim,
                           arch_style=arch_style,
                           model_path=f"../models/{data_name}-{img_resize}-{preprocess}/{arch_style}-{z_dim}/", )

    network.restore_model()

    results_path = f"../results/{data_name}-{img_resize}-{preprocess}/{arch_style}-{z_dim}/{source_key} to {target_key}/"
    os.makedirs(results_path, exist_ok=True)

    if sparse.issparse(valid_data.X):
        valid_data.X = valid_data.X.A
    k = len(test_digits)
    for j in range(n_files):
        source_sample = []
        target_sample = []
        for digit in test_digits:
            source_images_digit = valid_data[
                (valid_data.obs['labels'] == digit) & (valid_data.obs['condition'] == source_key)]
            target_images_digit = valid_data[
                (valid_data.obs['labels'] == digit) & (valid_data.obs['condition'] == target_key)]
            if j == 0:
                source_images_digit.X /= 255.0
            random_samples = np.random.choice(source_images_digit.shape[0], 1, replace=False)

            source_sample.append(source_images_digit.X[random_samples])
            target_sample.append(target_images_digit.X[random_samples])
        # random_samples = np.random.choice(source_images.shape[0], k, replace=False)
        # source_sample = source_data.X[random_samples]
        source_sample = np.array(source_sample)
        target_sample = np.array(target_sample)

        source_sample = np.reshape(source_sample, (-1, np.prod(image_shape)))
        source_sample_reshaped = np.reshape(source_sample, (-1, *image_shape))
        target_sample_reshaped = np.reshape(target_sample, (-1, *image_shape))

        source_sample = anndata.AnnData(X=source_sample)
        source_sample.obs['condition'] = np.ones(shape=(k, 1))

        pred_sample = network.predict(data=source_sample,
                                      encoder_labels=np.zeros((k, 1)),
                                      decoder_labels=np.ones((k, 1)))
        pred_sample = np.reshape(pred_sample, newshape=(-1, *image_shape))

        print(source_sample.shape, source_sample_reshaped.shape, target_sample_reshaped.shape, pred_sample.shape)

        plt.close("all")
        fig, ax = plt.subplots(k, 3, figsize=(k * 1, 6))
        for i in range(k):
            ax[i, 0].axis('off')
            if source_sample_reshaped.shape[-1] > 1:
                ax[i, 0].imshow(source_sample_reshaped[i])
            else:
                ax[i, 0].imshow(source_sample_reshaped[i, :, :, 0], cmap='Greys')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
            # if i == 0:
            #     if data_name == "celeba":
            #         ax[i, 0].set_title("Male without Eyeglasses")
            #         ax[i, 1].set_title("Male with Eyeglasses")
            if pred_sample.shape[-1] > 1:
                ax[i, 1].imshow(pred_sample[i])
            else:
                ax[i, 1].imshow(pred_sample[i, :, :, 0], cmap='Greys')

            ax[i, 2].imshow(target_sample_reshaped[i, :, :, 0], cmap='Greys')
        plt.savefig(os.path.join(results_path, f"./sample_images_{data_name}_{j}.pdf"))


def visualize_trained_network_results(data_dict, z_dim=100, arch_style=1, preprocess=True):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_size = data_dict.get('size', None)
    img_resize = data_dict.get('resize', None)
    n_channels = data_dict.get('n_channels', None)
    train_digits = data_dict.get('train_digits', None)
    test_digits = data_dict.get('test_digits', None)
    attribute = data_dict.get('attribute', None)

    path_to_save = f"../results/{data_name}-{img_resize}-{preprocess}/{arch_style}-{z_dim}/{source_key} to {target_key}/UMAPs/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        data = rcvae.prepare_and_load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                             attr_path="../data/celeba/list_attr_celeba.txt",
                                             gender=gender,
                                             attribute=attribute,
                                             max_n_images=5000,
                                             img_resize=img_resize,
                                             restore=True,
                                             save=False)

        if sparse.issparse(data.X):
            data.X = data.X.A

        train_images = data.X
        train_labels, _ = rcvae.label_encoder(data)

        train_data = anndata.AnnData(X=data)
        train_data.obs['condition'] = train_labels
        train_data.obs['labels'] = data.obs['labels'].values

        if preprocess:
            train_images /= 255.0
    else:
        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        train_images = np.reshape(train_data.X, (-1, img_size, img_size, n_channels))

        if img_resize != img_size:
            train_images = rcvae.resize_image(train_images, img_resize)
            train_images = np.reshape(train_images, (-1, img_resize, img_resize, n_channels))

        if preprocess:
            train_images /= 255.0

    train_labels, _ = rcvae.label_encoder(train_data)
    fake_labels = np.ones(train_labels.shape)

    network = rcvae.RCCVAE(x_dimension=(img_resize, img_resize, n_channels),
                           z_dimension=z_dim,
                           arch_style=arch_style,
                           model_path=f"../models/{data_name}-{img_resize}-{preprocess}/{arch_style}-{z_dim}/", )

    network.restore_model()

    train_data_feed = np.reshape(train_images, (-1, img_resize, img_resize, n_channels))

    latent_with_true_labels = network.to_latent(train_data_feed, train_labels)
    latent_with_fake_labels = network.to_latent(train_data_feed, fake_labels)
    mmd_latent_with_true_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=False)
    mmd_latent_with_fake_labels = network.to_mmd_layer(network, train_data_feed, train_labels, feed_fake=True)

    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs['condition'] = train_data.obs['condition'].values

    latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
    latent_with_fake_labels.obs['condition'] = train_data.obs['condition'].values

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs['condition'] = train_data.obs['condition'].values

    mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
    mmd_latent_with_fake_labels.obs['condition'] = train_data.obs['condition'].values

    if data_name.__contains__("mnist") or data_name == "celeba":
        latent_with_true_labels.obs['labels'] = train_data.obs['labels']
        latent_with_fake_labels.obs['labels'] = train_data.obs['labels']
        mmd_latent_with_true_labels.obs['labels'] = train_data.obs['labels']
        mmd_latent_with_fake_labels.obs['labels'] = train_data.obs['labels']

        color = ['condition', 'labels']
    else:
        color = ['condition']

    train_data.obs.loc[(train_data.obs['condition'] == source_key) & (
        train_data.obs['labels'].isin(train_digits)), 'type'] = 'training'
    train_data.obs.loc[
        (train_data.obs['condition'] == source_key) & (train_data.obs['labels'].isin(test_digits)), 'type'] = 'training'
    train_data.obs.loc[(train_data.obs['condition'] == target_key) & (
        train_data.obs['labels'].isin(train_digits)), 'type'] = 'training'
    train_data.obs.loc[
        (train_data.obs['condition'] == target_key) & (train_data.obs['labels'].isin(test_digits)), 'type'] = 'heldout'
    print(train_data.obs['labels'].value_counts())
    sc.pp.neighbors(train_data)
    sc.tl.umap(train_data)
    sc.pl.umap(train_data, color=color,
               save=f'_{data_name}_train_data.png',
               show=False,
               wspace=0.5)

    sc.tl.umap(train_data)
    sc.pl.umap(train_data, color=['type'],
               save=f'_{data_name}_data_type.png',
               show=False)

    sc.pp.neighbors(latent_with_true_labels)
    sc.tl.umap(latent_with_true_labels)
    sc.pl.umap(latent_with_true_labels, color=color,
               save=f"_{data_name}_latent_with_true_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(latent_with_fake_labels)
    sc.tl.umap(latent_with_fake_labels)
    sc.pl.umap(latent_with_fake_labels, color=color,
               save=f"_{data_name}_latent_with_fake_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(mmd_latent_with_true_labels)
    sc.tl.umap(mmd_latent_with_true_labels)
    sc.pl.umap(mmd_latent_with_true_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_true_labels.png",
               wspace=0.5,
               show=False)

    sc.pp.neighbors(mmd_latent_with_fake_labels)
    sc.tl.umap(mmd_latent_with_fake_labels)
    sc.pl.umap(mmd_latent_with_fake_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_fake_labels.png",
               wspace=0.5,
               show=False)

    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=20, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-m', '--mmd_dimension', type=int, default=128, required=False,
                                 help='MMD Layer dimension')
    arguments_group.add_argument('-a', '--alpha', type=float, default=0.005, required=False,
                                 help='Alpha coeff in loss term')
    arguments_group.add_argument('-b', '--beta', type=float, default=100, required=False,
                                 help='Beta coeff in loss term')
    arguments_group.add_argument('-k', '--kernel', type=str, default='multi-scale-rbf', required=False,
                                 help='Kernel type')
    arguments_group.add_argument('-n', '--n_epochs', type=int, default=5000, required=False,
                                 help='Maximum Number of epochs for training')
    arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                                 help='Batch Size')
    arguments_group.add_argument('-s', '--arch_style', type=int, default=1, required=False,
                                 help='Model Architecture Style')
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.4, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-e', '--resize', type=int, default=64, required=False,
                                 help='Image size to be resize')
    arguments_group.add_argument('-p', '--preprocess', type=int, default=True, required=False,
                                 help='do preprocess images')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning Rate for Optimizer')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    data_dict['resize'] = args['resize']
    del args['data']
    del args['resize']
    if args['preprocess'] == 0:
        args['preprocess'] = False
    else:
        args['preprocess'] = True
    train_network(data_dict=data_dict, **args)
    evaluate_network(data_dict,
                     z_dim=args['z_dim'],
                     n_files=30,
                     arch_style=args['arch_style'],
                     k=4)
    visualize_trained_network_results(data_dict,
                                      z_dim=args['z_dim'],
                                      arch_style=args['arch_style'],
                                      preprocess=args['preprocess'])
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
