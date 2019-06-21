import logging
import os

import keras
import numpy as np
from keras.callbacks import CSVLogger, History, EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D, \
    Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import multi_gpu_model, to_categorical
from scipy import sparse

from .utils import label_encoder

log = logging.getLogger(__file__)


class FaceNet:
    """
        Regularized Convolutional C-VAE vector Network class. This class contains the implementation of Conditional
        Variational Auto-encoder network.
        # Parameters
            kwargs:
                key: `dropout_rate`: float
                        dropout rate
                key: `learning_rate`: float
                    learning rate of optimization algorithm
                key: `model_path`: basestring
                    path to save the model after training
                key: `alpha`: float
                    alpha coefficient for loss.
                key: `beta`: float
                    beta coefficient for loss.
            x_dimension: integer
                number of gene expression space dimensions.
            z_dimension: integer
                number of latent space dimensions.
    """

    def __init__(self, x_dimension, z_dimension=100, **kwargs):
        self.x_dim = x_dimension if isinstance(x_dimension, tuple) else (x_dimension,)
        self.z_dim = z_dimension
        self.image_shape = x_dimension

        self.lr = kwargs.get("learning_rate", 0.001)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "../models/")
        self.n_gpus = kwargs.get("gpus", 1)

        self.x = Input(shape=self.x_dim, name="image")

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function(compile_gpu_model=True)
        self.model.summary()

    def _create_network(self):
        """
            Constructs the whole C-VAE network. It is step-by-step constructing the C-VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of C-VAE.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """
        conv = Conv2D(8, 6, padding='same', kernel_initializer='he_normal')(self.x)
        conv = LeakyReLU(name='conv1')(conv)
        max_pooling = MaxPooling2D(2)(conv)

        conv = Conv2D(16, 4, padding='same', kernel_initializer='he_normal')(max_pooling)
        conv = LeakyReLU(name='conv4')(conv)
        max_pooling = MaxPooling2D(2)(conv)

        conv = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(max_pooling)
        conv = LeakyReLU(name='conv7')(conv)
        max_pooling = MaxPooling2D(2)(conv)

        flat = Flatten()(max_pooling)

        dense = Dense(512, kernel_initializer='he_normal')(flat)
        dense = BatchNormalization()(dense)
        dense = LeakyReLU()(dense)
        dense = Dropout(self.dr_rate)(dense)

        dense = Dense(4, activation='softmax', kernel_initializer='he_normal')(dense)

        self.model = Model(inputs=self.x, outputs=dense)

        if self.n_gpus > 1:
            self.gpu_model = multi_gpu_model(self.model, gpus=self.n_gpus)
        else:
            self.gpu_model = self.model

    def _loss_function(self, compile_gpu_model=True):
        """
            Defines the loss function of C-VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            C-VAE and also defines the Optimization algorithm for network. The C-VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
        """

        self.model_optimizer = keras.optimizers.Adam(lr=self.lr)
        if compile_gpu_model:
            self.gpu_model.compile(optimizer=self.model_optimizer,
                                   loss='categorical_crossentropy',
                                   metrics=['acc'])
        else:
            self.model.compile(optimizer=self.model_optimizer,
                               loss='categorical_crossentropy',
                               metrics=['acc'])

    def train(self, train_data, use_validation=False, valid_data=None, n_epochs=25, batch_size=32, early_stop_limit=20,
              threshold=0.0025, initial_run=True,
              shuffle=True, verbose=2, save=True):  # TODO: Write minibatches for each source and destination
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent overfitting.
            # Parameters
                n_epochs: int
                    number of epochs to iterate and optimize network weights
                early_stop_limit: int
                    number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                full_training: bool
                    if `True`: Network will be trained with all batches of data in each epoch.
                    if `False`: Network will be trained with a random batch of data in each epoch.
                initial_run: bool
                    if `True`: The network will initiate training and log some useful initial messages.
                    if `False`: Network will resume the training using `restore_model` function in order
                        to restore last model which has been trained with some training dataset.
            # Returns
                Nothing will be returned
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read(train_katrain_kang.h5ad           >>> validation_data = sc.read(valid_kang.h5ad)
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            ```
        """
        if initial_run:
            log.info("----Training----")
        train_labels = to_categorical(train_data.obs['label'].values)

        if sparse.issparse(train_data.X):
            train_data.X = train_data.X.A

        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        callbacks = [
            History(),
            EarlyStopping(patience=early_stop_limit, monitor='val_loss', min_delta=threshold),
            CSVLogger(filename="./facenet_train.log")
        ]

        x_train = np.reshape(train_data.X, newshape=(-1, *self.x_dim))
        x = x_train
        y = train_labels

        if use_validation:
            x_valid = np.reshape(valid_data.X, newshape=(-1, *self.x_dim))
            valid_labels = to_categorical(valid_data.obs['label'].values)
            x_test = x_valid
            y_test = valid_labels
            print(x.shape, y.shape)
            print(x_test.shape, y_test.shape)
            histories = self.gpu_model.fit(x=x,
                                           y=y,
                                           epochs=n_epochs,
                                           batch_size=batch_size,
                                           validation_data=(x_test, y_test),
                                           shuffle=shuffle,
                                           callbacks=callbacks,
                                           verbose=verbose)
        else:
            print(x.shape, y.shape)
            histories = self.gpu_model.fit(x=x,
                                           y=y,
                                           epochs=n_epochs,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           callbacks=callbacks,
                                           verbose=verbose)
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.model.save(os.path.join(self.model_to_use, "facenet_model.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
        return histories
