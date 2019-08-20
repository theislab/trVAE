import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import CSVLogger, History, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.layers import Activation
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Conv2D, \
    Flatten, Reshape, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import multi_gpu_model, to_categorical

from trvae.models._utils import sample_z, print_message
from trvae.utils import label_encoder, remove_sparsity
from ._losses import LOSSES

log = logging.getLogger(__file__)


class DCtrVAE:
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

        self.n_conditions = kwargs.get("n_conditions", 2)
        self.mmd_dim = kwargs.get("mmd_dimension", 128)
        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./")
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.arch_style = kwargs.get("arch_style", 1)
        self.n_gpus = kwargs.get("gpus", 1)

        self.x = Input(shape=self.x_dim, name="data")
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function(compile_gpu_model=True)
        self.cvae_model.summary()

    def _encoder(self, name="encoder"):
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
        if self.arch_style == 1:  # Baseline CNN
            h = Dense(128, activation='relu')(self.encoder_labels)
            h = Dense(np.prod(self.x_dim[:-1]), activation='relu')(h)
            h = Reshape((*self.x_dim[:-1], 1))(h)
            h = concatenate([self.x, h])
            h = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', activation=None)(h)
            h = LeakyReLU()(h)
            h = MaxPooling2D()(h)
            h = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', activation=None)(h)
            h = LeakyReLU()(h)
            h = MaxPooling2D()(h)
            h = Flatten()(h)
            h = Dense(1024, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            model.summary()
            return mean, log_var, model
        elif self.arch_style == 2:  # FCN
            x_reshaped = Reshape(target_shape=(np.prod(self.x_dim),))(self.x)
            xy = concatenate([x_reshaped, self.encoder_labels], axis=1)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(xy)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
            z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            model.summary()
            return mean, log_var, model
        else:
            h = Dense(128, activation='relu')(self.encoder_labels)
            h = Dense(np.prod(self.x_dim[:-1]), activation='relu')(h)
            h = Reshape((*self.x_dim[:-1], 1))(h)
            h = concatenate([self.x, h])

            conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_1')(h)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_2')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_2')(conv3)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_3')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_1')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_2')(conv4)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', name='conv4_3')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_1')(pool4)
            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_2')(conv5)
            conv5 = Conv2D(512, 3, activation='relu', padding='same', name='conv5_3')(conv5)
            pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

            flat = Flatten(name='flatten')(pool5)

            dense = Dense(1024, activation='linear', name='fc1')(flat)
            dense = Activation('relu')(dense)

            dense = Dense(256, activation='linear', name='fc2')(dense)
            dense = Activation('relu')(dense)
            self.enc_dense = Dropout(self.dr_rate)(dense)

            mean = Dense(self.z_dim, kernel_initializer=self.init_w)(self.enc_dense)
            log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(self.enc_dense)

            z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
            model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
            model.summary()
            return mean, log_var, model

    def _mmd_decoder(self, name="decoder"):
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
        if self.arch_style == 1:  # Baseline CNN for MNIST
            zy = concatenate([self.z, self.decoder_labels], axis=1)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization(axis=1)(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dropout(self.dr_rate)(h_mmd)
            h = Dense(1024, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(np.prod(self.x_dim), kernel_initializer=self.init_w, use_bias=False)(h)
            h = LeakyReLU()(h)
            h = Reshape(target_shape=self.x_dim)(h)
            h = Conv2DTranspose(128, kernel_size=(4, 4), padding='same')(h)
            h = LeakyReLU()(h)
            h = Conv2DTranspose(64, kernel_size=(4, 4), padding='same')(h)
            h = LeakyReLU()(h)
            h = Conv2DTranspose(self.x_dim[-1], kernel_size=(4, 4), padding='same', activation="relu")(h)
            decoder_model = Model(inputs=[self.z, self.decoder_labels], outputs=h, name=name)
            decoder_mmd_model = Model(inputs=[self.z, self.decoder_labels], outputs=h_mmd, name='deocder_mmd')
            return decoder_model, decoder_mmd_model

        elif self.arch_style == 2:  # FCN
            zy = concatenate([self.z, self.decoder_labels], axis=1)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization(axis=1)(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(np.prod(self.x_dim), kernel_initializer=self.init_w, use_bias=True)(h)
            h = Activation('relu', name="reconstruction_output")(h)
            h = Reshape(target_shape=self.x_dim)(h)
            decoder_model = Model(inputs=[self.z, self.decoder_labels], outputs=h, name=name)
            decoder_mmd_model = Model(inputs=[self.z, self.decoder_labels], outputs=h_mmd, name='deocder_mmd')
            return decoder_model, decoder_mmd_model
        else:
            encode_y = Dense(64, activation='relu')(self.decoder_labels)
            zy = concatenate([self.z, encode_y], axis=1)
            zy = Activation('relu')(zy)

            h = Dense(self.mmd_dim, activation="linear", kernel_initializer='he_normal')(zy)
            h_mmd = Activation('relu', name="mmd")(h)

            h = Dense(1024, kernel_initializer='he_normal')(h_mmd)
            h = Activation('relu')(h)

            h = Dense(256 * 4 * 4, kernel_initializer='he_normal')(h)
            h = Activation('relu')(h)

            width = self.x_dim[0] // 16
            height = self.x_dim[1] // 16
            n_channels = 4096 // (width * height)
            h = Reshape(target_shape=(width, height, n_channels))(h)

            up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(h))
            conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)

            up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)

            up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)

            up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv8))
            conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)

            conv10 = Conv2D(self.x_dim[-1], 1, activation='relu')(conv9)

            decoder_model = Model(inputs=[self.z, self.decoder_labels], outputs=conv10, name=name)
            decoder_mmd_model = Model(inputs=[self.z, self.decoder_labels], outputs=h_mmd, name='deocder_mmd')
            return decoder_model, decoder_mmd_model

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

        inputs = [self.x, self.encoder_labels, self.decoder_labels]
        self.mu, self.log_var, self.encoder_model = self._encoder(name="encoder")
        self.decoder_model, self.decoder_mmd_model = self._mmd_decoder(name="decoder")

        decoder_output = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        mmd_output = self.decoder_mmd_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])

        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_output)
        mmd_output = Lambda(lambda x: x, name="mmd")(mmd_output)

        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")
        if self.n_gpus > 1:
            self.gpu_cvae_model = multi_gpu_model(self.cvae_model,
                                                  gpus=self.n_gpus)
            self.gpu_encoder_model = multi_gpu_model(self.encoder_model,
                                                     gpus=self.n_gpus)
            self.gpu_decoder_model = multi_gpu_model(self.decoder_model,
                                                     gpus=self.n_gpus)
        else:
            self.gpu_cvae_model = self.cvae_model
            self.gpu_encoder_model = self.encoder_model
            self.gpu_decoder_model = self.decoder_model

    def _calculate_loss(self):
        loss = LOSSES['mse'](self.mu, self.log_var, self.alpha, self.eta)
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta, self.kernel_method, "general")
        return loss, mmd_loss

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

        mse_loss, mmd_loss = self._calculate_loss()
        self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
        if compile_gpu_model:
            self.gpu_cvae_model.compile(optimizer=self.cvae_optimizer,
                                        loss=[mse_loss, mmd_loss],
                                        metrics={self.cvae_model.outputs[0].name: mse_loss,
                                                 self.cvae_model.outputs[1].name: mmd_loss})
        else:
            self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                    loss=[mse_loss, mmd_loss],
                                    metrics={self.cvae_model.outputs[0].name: mse_loss,
                                             self.cvae_model.outputs[1].name: mmd_loss})

    def to_latent(self, adata, encoder_labels):
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
        adata = remove_sparsity(adata)

        images = np.reshape(adata.X, (-1, *self.x_dim))
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)

        latent = self.encoder_model.predict([images, encoder_labels])[2]

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def to_mmd_layer(self, adata, encoder_labels, feed_fake=0):
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
        if feed_fake > 0:
            decoder_labels = np.zeros(shape=encoder_labels.shape) + feed_fake
        else:
            decoder_labels = encoder_labels
        adata = remove_sparsity(adata)

        images = np.reshape(adata.X, (-1, *self.x_dim))
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        mmd_latent = self.cvae_model.predict([images, encoder_labels, decoder_labels])[1]
        mmd_adata = anndata.AnnData(X=mmd_latent)
        mmd_adata.obs = adata.obs.copy(deep=True)

        return mmd_adata

    def predict(self, adata, encoder_labels, decoder_labels):
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
        adata = remove_sparsity(adata)

        images = np.reshape(adata.X, (-1, *self.x_dim))
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        reconstructed = self.cvae_model.predict([images, encoder_labels, decoder_labels])[0]
        reconstructed = np.reshape(reconstructed, (-1, np.prod(self.x_dim)))

        reconstructed_adata = anndata.AnnData(X=reconstructed)
        reconstructed_adata.obs = adata.obs.copy(deep=True)
        reconstructed_adata.var_names = adata.var_names

        return reconstructed_adata

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
        self.cvae_model = load_model(os.path.join(self.model_path, 'mmd_cvae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_path, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_path, 'decoder.h5'), compile=False)
        self.decoder_mmd_model = load_model(os.path.join(self.model_path, 'decoder_mmd.h5'), compile=False)
        self._loss_function(compile_gpu_model=False)

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_path, "mmd_cvae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_path, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_path, "decoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_path, "decoder_mmd.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_path}. Training finished")

    def train(self, train_adata, valid_adata=None,
              condition_encoder=None, condition_key='condition',
              n_epochs=25, batch_size=32,
              early_stop_limit=20, lr_reducer=10, threshold=0.0025, monitor='val_loss',
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
        train_adata = remove_sparsity(train_adata)

        train_labels_encoded, self.condition_encoder = label_encoder(train_adata, condition_encoder, condition_key)
        train_labels_onehot = to_categorical(train_labels_encoded, num_classes=self.n_conditions)

        callbacks = [
            History(),
            CSVLogger(filename="./csv_logger.log"),
        ]
        if early_stop_limit > 0:
            callbacks.append(EarlyStopping(patience=early_stop_limit, monitor=monitor, min_delta=threshold))

        if lr_reducer > 0:
            callbacks.append(ReduceLROnPlateau(monitor=monitor, patience=lr_reducer, verbose=verbose))

        if verbose > 2:
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print_message(epoch, logs, n_epochs, verbose)))
            fit_verbose = 0
        else:
            fit_verbose = verbose

        train_images = np.reshape(train_adata.X, (-1, *self.x_dim))

        x = [train_images, train_labels_onehot, train_labels_onehot]
        y = [train_images, train_labels_encoded]

        if valid_adata is not None:
            valid_adata = remove_sparsity(valid_adata)

            valid_labels_encoded, _ = label_encoder(valid_adata, condition_encoder, condition_key)
            valid_labels_onehot = to_categorical(valid_labels_encoded, num_classes=self.n_conditions)

            valid_images = np.reshape(valid_adata.X, (-1, *self.x_dim))

            x_valid = [valid_images, valid_labels_onehot, valid_labels_onehot]
            y_valid = [valid_images, valid_labels_encoded]

            self.cvae_model.fit(x=x,
                                y=y,
                                epochs=n_epochs,
                                batch_size=batch_size,
                                validation_data=(x_valid, y_valid),
                                shuffle=shuffle,
                                callbacks=callbacks,
                                verbose=fit_verbose)
        else:
            self.cvae_model.fit(x=x,
                                y=y,
                                epochs=n_epochs,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                callbacks=callbacks,
                                verbose=fit_verbose)
        if save:
            self.save_model()

