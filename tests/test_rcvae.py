import os

import anndata
import numpy as np

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt


def train_celeba(z_dim=100,
                 mmd_dimension=256,
                 alpha=0.001,
                 beta=100,
                 kernel='multi-scale-rbf',
                 n_epochs=500,
                 batch_size=512,
                 dropout_rate=0.2,
                 arch_style=1,
                 ):
    source_images, target_images = rcvae.load_celeba(file_path="../data/celebA/img_align_celeba.zip",
                                                     attr_path="../data/celebA/list_attr_celeba.txt",
                                                     max_n_images=50000,
                                                     gender='Male', source_attr='Black_Hair', target_attr='Blond_Hair',
                                                     img_resize=32,
                                                     restore=False,
                                                     save=True)

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
                           model_path="../models/",
                           dropout_rate=dropout_rate)

    network.train(train_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=100,
                  shuffle=True,
                  save=True)

    print("Model has been trained")


def evaluate_network(data_name="celeba"):
    source_images, target_images = rcvae.load_celeba(file_path="../data/celebA/img_align_celeba.zip",
                                                     attr_path="../data/celebA/list_attr_celeba.txt",
                                                     gender='Male', source_attr='Black_Hair', target_attr='Blond_Hair',
                                                     max_n_images=5000,
                                                     img_resize=32,
                                                     restore=True,
                                                     save=False)
    if data_name == "celeba":
        img_size = 32
        n_channels = 3
    else:
        img_size = 28
        n_channels = 1

    image_shape = (img_size, img_size, n_channels)

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
                           model_path="../models/")

    network.restore_model()

    if data_name == "celeba":
        results_path = f"../results/{data_name}/On Hair/"
        os.makedirs(results_path, exist_ok=True)
        os.chdir(results_path)

    for j in range(5):
        k = 5
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
            ax[i, 0].imshow(source_sample_reshaped[i])
            ax[i, 1].axis('off')
            if i == 0:
                if data_name == "celeba":
                    ax[i, 0].set_title("Male without Eyeglasses")
                    ax[i, 1].set_title("Male with Eyeglasses")

            ax[i, 1].imshow(target_sample[i])
        plt.savefig(f"./sample_images_{data_name}_{j}.pdf")


if __name__ == '__main__':
    train_celeba(z_dim=100,
                 mmd_dimension=256,
                 alpha=0.001,
                 beta=1000,
                 kernel='multi-scale-rbf',
                 n_epochs=1000,
                 batch_size=512,
                 arch_style=3,
                 dropout_rate=0.25)
    evaluate_network("celeba")
