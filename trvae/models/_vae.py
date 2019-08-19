import logging
import os

import keras
import numpy as np
from keras import backend as K
from keras.callbacks import CSVLogger, History, EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout, Input, Lambda, Reshape, Conv1D, \
    MaxPooling1D, Flatten, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from scipy import sparse

from trvae.models._utils import sample_z

log = logging.getLogger(__file__)


class VAE:
    """
        VAE Network class. This class contains the implementation of
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
        self.x_dim = x_dimension
        self.z_dim = z_dimension

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.conditions = kwargs.get("condition_list")
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.n_gpus = kwargs.get("gpus", 1)
        self.arch_style = kwargs.get("arch_style", 1)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function(compile_gpu_model=True)

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.vae_model.summary()

    def _encoder(self, x, name="encoder"):
        """
            Constructs the encoder sub-network of C-VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.
            # Parameters
                No parameters are needed.
            # Returns
                mean: Tensor
                    A dense layer consists of means of gaussian distributions of latent space dimensions.
                log_var: Tensor
                    A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        if self.arch_style == 1:
            h = Reshape((self.x_dim, 1))(x)

            h = Conv1D(32, kernel_size=256, activation='relu', padding='valid')(h)
            # h = Conv1D(32, kernel_size=256, activation='relu', padding='same')(h)
            h = MaxPooling1D(pool_size=100)(h)

            h = Flatten()(h)

            h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
        elif self.arch_style == 2:
            h = Dense(4096, kernel_initializer=self.init_w, use_bias=False)(x)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

            h = Dense(self.z_dim, kernel_initializer=self.init_w)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])

        model = Model(inputs=x, outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _decoder(self, z, name="decoder"):
        """
            Constructs the decoder sub-network of C-VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.
            # Parameters
                No parameters are needed.
            # Returns
                h: Tensor
                    A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.
        """
        if self.arch_style == 1:
            h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(z)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)

            h = Reshape((256, 1, 1))(h)

            h = UpSampling2D(size=(16, 1))(h)
            # h = Conv2DTranspose(32, kernel_size=(256, 1), activation='relu', padding='same',
            #                     kernel_initializer='he_normal')(h)
            # h = Conv2DTranspose(64, kernel_size=(512, 1), activation='relu', padding='same', kernel_initializer='he_normal')(h)
            # h = Conv2DTranspose(256, kernel_size=(1024, 1), activation='relu', padding='same', kernel_initializer='he_normal')(h)

            # h = UpSampling2D(size=(2, 1))(h)
            # h = Conv2DTranspose(64, kernel_size=(256, 1), activation='relu', padding='same',
            #                     kernel_initializer='he_normal')(h)
            # h = Conv2DTranspose(256, kernel_size=(1024, 1), activation='relu', padding='same', kernel_initializer='he_normal')(h)
            # h = Conv2DTranspose(256, kernel_size=(1024, 1), activation='relu', padding='same')(h)

            # h = UpSampling2D(size=(4, 1))(h)
            # h = Conv2DTranspose(4, kernel_size=(3152, 1), activation='relu', padding='valid')(h)
            h = Conv2DTranspose(32, kernel_size=(905, 1), activation='relu', padding='valid')(h)
            h = Conv2DTranspose(1, kernel_size=(256, 1), activation='relu', padding='same')(h)
            h = Reshape((self.x_dim,))(h)

        if self.arch_style == 2:
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(z)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

            h = Dense(4096, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

            h = Dense(self.x_dim, activation='relu', kernel_initializer=self.init_w, use_bias=True)(h)

        model = Model(inputs=z, outputs=h, name=name)
        return h, model

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

        self.mu, self.log_var, self.encoder_model = self._encoder(self.x, name="encoder")
        self.x_hat, self.decoder_model = self._decoder(self.z, name="decoder")
        decoder_outputs = self.decoder_model(self.encoder_model(self.x)[2])
        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_outputs)
        self.vae_model = Model(inputs=self.x,
                               outputs=reconstruction_output,
                               name="vae")

        if self.n_gpus > 1:
            self.gpu_vae_model = multi_gpu_model(self.vae_model,
                                                 gpus=self.n_gpus)
            self.gpu_encoder_model = multi_gpu_model(self.encoder_model,
                                                     gpus=self.n_gpus)
            self.gpu_decoder_model = multi_gpu_model(self.decoder_model,
                                                     gpus=self.n_gpus)
        else:
            self.gpu_vae_model = self.vae_model
            self.gpu_encoder_model = self.encoder_model
            self.gpu_decoder_model = self.decoder_model

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

        def kl_recon_loss(y_true, y_pred):
            kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)
            # recon_loss = 0.5 * K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
            recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
            # output_shape = K.shape(y_pred)
            #
            # def get_column(tensor, col):
            #     with tf.variable_scope("gather_cols", reuse=tf.AUTO_REUSE):
            #         tensor = K.tf.convert_to_tensor(tensor, name='tensor')
            #         col = K.tf.convert_to_tensor(col, name='column')
            #
            #         return tf.transpose(tf.nn.embedding_lookup(tf.transpose(tensor), col))
            #
            # def body(i):
            #     y_true_i = get_column(y_true, i)
            #     y_pred_i = get_column(y_pred, i)
            #     bc_loss = K.binary_crossentropy(y_true_i, y_pred_i)
            #     print(bc_loss)
            #     return bc_loss

            # recon_loss = K.sum(K.map_fn(body, K.arange(0, self.x_dim)), axis=0)
            # recon_loss = K.cast(recon_loss, dtype='float32')
            # print(recon_loss)
            # print(kl_loss)
            return recon_loss + self.alpha * kl_loss

        self.vae_optimizer = keras.optimizers.Adam(lr=self.lr)
        if compile_gpu_model:
            self.gpu_vae_model.compile(optimizer=self.vae_optimizer,
                                       loss=kl_recon_loss,
                                       metrics={self.vae_model.outputs[0].name: kl_recon_loss})
        else:
            self.vae_model.compile(optimizer=self.vae_optimizer,
                                   loss=kl_recon_loss,
                                   metrics={self.vae_model.outputs[0].name: kl_recon_loss})

    def to_latent(self, data):
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        latent = self.encoder_model.predict(data)[2]
        return latent

    def to_mmd_layer(self, model, data, encoder_labels, feed_fake=False):
        """
            Map `data` in to the pn layer after latent layer. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        if feed_fake:
            decoder_labels = np.ones(shape=encoder_labels.shape)
        else:
            decoder_labels = encoder_labels
        mmd_latent = model.cvae_model.predict([data, encoder_labels, decoder_labels])[1]
        return mmd_latent

    def _reconstruct(self, data, use_data=False):
        """
            Map back the latent space encoding via the decoder.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in latent space or primary space.
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                use_data: bool
                    this flag determines whether the `data` is already in latent space or not.
                    if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                    if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).
            # Returns
                rec_data: 'numpy nd-array'
                    returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        if use_data:
            latent = data
        else:
            latent = self.to_latent(data)
        rec_data = self.decoder_model.predict(latent)
        return rec_data

    def predict(self, data, data_space='None'):
        """
            Predicts the cell type provided by the user in stimulated condition.
            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
            # Returns
                stim_pred: numpy nd-array
                    `numpy nd-array` of predicted cells in primary space.
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("train_kang.h5ad")
            validation_data = sc.read("./data/validation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            prediction = network.predict('CD4T', obs_key={"cell_type": ["CD8T", "NK"]})
            ```
        """
        if sparse.issparse(data.X):
            if data_space == 'latent':
                stim_pred = self._reconstruct(data.X.A, use_data=True)
            else:
                stim_pred = self._reconstruct(data.X.A)
        else:
            if data_space == 'latent':
                stim_pred = self._reconstruct(data.X, use_data=True)
            else:
                stim_pred = self._reconstruct(data.X)
        return stim_pred[0]

    def restore_model(self):
        """
            restores model weights from `model_to_use`.
            # Parameters
                No parameters are needed.
            # Returns
                Nothing will be returned.
            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("./data/train_kang.h5ad")
            validation_data = sc.read("./data/valiation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.restore_model()
            ```
        """
        self.vae_model = load_model(os.path.join(self.model_to_use, 'vae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function()

    def train(self, train_data, use_validation=False, valid_data=None, n_epochs=25, batch_size=32, early_stop_limit=20,
              threshold=0.0025, initial_run=True,
              shuffle=True, verbose=2, save=True):
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

        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        callbacks = [
            History(),
            EarlyStopping(patience=early_stop_limit, monitor='val_loss', min_delta=threshold),
            CSVLogger(filename="./csv_logger.log")
        ]

        if sparse.issparse(train_data.X):
            train_data.X = train_data.X.A

        x = train_data.X
        y = train_data.X
        if use_validation:
            if sparse.issparse(valid_data.X):
                valid_data.X = valid_data.X.A

            x_valid = valid_data.X
            y_valid = valid_data.X
            histories = self.gpu_vae_model.fit(
                x=x,
                y=y,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        else:
            histories = self.gpu_vae_model.fit(
                x=x,
                y=y,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.vae_model.save(os.path.join(self.model_to_use, "vae.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
        return histories
