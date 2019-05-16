import os

import anndata
import numpy as np

import rcvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")


# from datetime import datetime, timezone

# current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
# os.makedirs(current_time, exist_ok=True)
# os.chdir("./" + current_time)

def train_celeba(z_dim=50,
                 alpha=0.001,
                 beta=100,
                 kernel='multi-scale-rbf',
                 n_epochs=500,
                 batch_size=1024,
                 dropout_rate=0.2,
                 ):
    source_images, target_images = rcvae.load_celeba(file_path="../data/celebA/img_align_celeba.zip",
                                                     attr_path="../data/celebA/list_attr_celeba.txt",
                                                     max_n_images=30000,
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
                           alpha=alpha,
                           beta=beta,
                           kernel=kernel,
                           train_with_fake_labels=True,
                           model_path="../models/",
                           dropout_rate=dropout_rate)

    network.train(train_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size)


if __name__ == '__main__':
    train_celeba()
