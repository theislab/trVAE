import argparse
import os

import anndata
import numpy as np
import scanpy as sc

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
    "CelebA": {"name": 'celeba', "gender": "Male", "source_key": "Wearing_Hat", "target_key": "Wearing_Hat",
               "resize": 64, "n_channels": 3},
    "MNIST": {"name": 'mnist', "source_key": 1, "target_key": 7, "resize": 28, 'size': 28, "n_channels": 1},
    "ThinMNIST": {"name": 'thin_mnist', "source_key": "normal", "target_key": "thin", "resize": 28, 'size': 28,
                  "n_channels": 1},
    "ThickMNIST": {"name": 'thick_mnist', "source_key": "normal", "target_key": "thick", "resize": 28, 'size': 28,
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
                  ):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_resize = data_dict.get("resize", None)
    n_channels = data_dict.get("n_channels", None)
    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        source_images, target_images = rcvae.load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                                         attr_path="../data/celeba/list_attr_celeba.txt",
                                                         max_n_images=50000,
                                                         gender=gender, source_attr=source_key, target_attr=target_key,
                                                         img_resize=img_resize,
                                                         restore=False,
                                                         save=True)
    else:
        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        img_size = data_dict.get("size", None)
        if isinstance(source_key, list):
            source_images = train_data[train_data.obs["condition"].isin(source_key)].X
            target_images = train_data[train_data.obs["condition"].isin(target_key)].X
        else:
            source_images = train_data[train_data.obs["condition"] == source_key].X
            target_images = train_data[train_data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_size, img_size, n_channels))
        target_images = np.reshape(target_images, (-1, img_size, img_size, n_channels))

        source_images = rcvae.resize_image(source_images, img_resize)
        target_images = rcvae.resize_image(target_images, img_resize)

        source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
        target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))

        source_images /= 255.0
        target_images /= 255.0

    source_labels = np.zeros(shape=source_images.shape[0])
    target_labels = np.ones(shape=target_images.shape[0])
    train_labels = np.concatenate([source_labels, target_labels], axis=0)

    train_images = np.concatenate([source_images, target_images], axis=0)
    train_images = np.reshape(train_images, (-1, np.prod(source_images.shape[1:])))

    train_data = anndata.AnnData(X=train_images)
    train_data.obs["condition"] = train_labels

    network = rcvae.RCCVAE(x_dimension=source_images.shape[1:],
                           z_dimension=z_dim,
                           mmd_dimension=mmd_dimension,
                           alpha=alpha,
                           beta=beta,
                           kernel=kernel,
                           arch_style=arch_style,
                           train_with_fake_labels=True,
                           model_path=f"../models/{data_name}/{arch_style}/",
                           dropout_rate=dropout_rate)

    network.train(train_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


def evaluate_network(data_dict=None, n_files=5, k=5, arch_style=1):
    data_name = data_dict['name']
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    img_resize = data_dict.get("resize", None)
    img_size = data_dict.get("size", None)
    n_channels = data_dict.get('n_channels', None)

    if data_name == "celeba":
        gender = data_dict.get('gender', None)
        source_images, target_images = rcvae.load_celeba(file_path="../data/celeba/img_align_celeba.zip",
                                                         attr_path="../data/celeba/list_attr_celeba.txt",
                                                         gender=gender, source_attr=source_key, target_attr=target_key,
                                                         max_n_images=5000,
                                                         img_resize=img_resize,
                                                         restore=True,
                                                         save=False)
    else:
        train_data = sc.read(f"../data/{data_name}/{data_name}.h5ad")
        if isinstance(source_key, list):
            source_images = train_data[train_data.obs["condition"].isin(source_key)].X
            target_images = train_data[train_data.obs["condition"].isin(target_key)].X
        else:
            source_images = train_data[train_data.obs["condition"] == source_key].X
            target_images = train_data[train_data.obs["condition"] == target_key].X

        source_images = np.reshape(source_images, (-1, img_size, img_size, n_channels))
        target_images = np.reshape(target_images, (-1, img_size, img_size, n_channels))

        source_images = rcvae.resize_image(source_images, img_resize)
        target_images = rcvae.resize_image(target_images, img_resize)

        source_images = np.reshape(source_images, (-1, img_resize, img_resize, n_channels))
        target_images = np.reshape(target_images, (-1, img_resize, img_resize, n_channels))

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
                           z_dimension=100,
                           model_path=f"../models/{data_name}/{arch_style}/", )

    network.restore_model()

    results_path = f"../results/{data_name}/{arch_style}/{source_key} to {target_key}/"
    os.makedirs(results_path, exist_ok=True)
    os.chdir(results_path)

    for j in range(n_files):
        random_samples = np.random.choice(source_images.shape[0], k, replace=False)

        source_sample = source_data.X[random_samples]
        source_sample_reshaped = np.reshape(source_sample, (-1, *image_shape))

        source_sample = anndata.AnnData(X=source_sample)
        source_sample.obs['condition'] = np.ones(shape=(k, 1))

        target_sample = network.predict(data=source_sample,
                                        encoder_labels=np.zeros((k, 1)),
                                        decoder_labels=np.ones((k, 1)))
        target_sample = np.reshape(target_sample, newshape=(-1, *image_shape))

        print(source_sample.shape, source_sample_reshaped.shape, target_sample.shape)

        plt.close("all")
        fig, ax = plt.subplots(k, 2, figsize=(k * 1, 6))
        for i in range(k):
            ax[i, 0].axis('off')
            if source_sample_reshaped.shape[-1] > 1:
                ax[i, 0].imshow(source_sample_reshaped[i])
            else:
                ax[i, 0].imshow(source_sample_reshaped[i, :, :, 0], cmap='Greys')
            ax[i, 1].axis('off')
            if i == 0:
                if data_name == "celeba":
                    ax[i, 0].set_title("Male without Eyeglasses")
                    ax[i, 1].set_title("Male with Eyeglasses")
            if target_sample.shape[-1] > 1:
                ax[i, 1].imshow(target_sample[i])
            else:
                ax[i, 1].imshow(target_sample[i, :, :, 0], cmap='Greys')
        plt.savefig(f"./sample_images_{data_name}_{j}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=100, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-m', '--mmd_dimension', type=int, default=128, required=False,
                                 help='MMD Layer dimension')
    arguments_group.add_argument('-a', '--alpha', type=float, default=0.001, required=False,
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
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.2, required=False,
                                 help='Dropout ratio')
    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    del args['data']
    train_network(data_dict=data_dict, **args)
    evaluate_network(data_dict,
                     n_files=30,
                     arch_style=args['arch_style'],
                     k=5)
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
