import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import CSVLogger, History, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import sparse

from trvae.models._losses import LOSSES
from trvae.models._activations import ACTIVATIONS
from trvae.models._utils import sample_z, print_message
from trvae.utils import label_encoder, remove_sparsity

log = logging.getLogger(__file__)


class MMDCVAE:
    """
        Regularized C-VAE vector Network class. This class contains the implementation of Conditional
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

    def __init__(self, x_dimension, z_dimension=100, n_conditions=3, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.n_conditions = n_conditions

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.mmd_computation_way = kwargs.get("mmd_computation_way", "general")
        self.ridge = kwargs.get('ridge', 0.1)
        self.scale_factor = kwargs.get('scale_factor', 1.0)
        self.clip_value = kwargs.get('clip_value', 3.0)
        self.lambda_l1 = kwargs.get('lambda_l1', 0.01)
        self.lambda_l2 = kwargs.get('lambda_l2', 0.1)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.encoder_labels = Input(shape=(self.n_conditions,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_conditions,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.init_w = keras.initializers.glorot_normal()
        self.regularizer = keras.regularizers.l1_l2(self.lambda_l1, self.lambda_l2)
        self._create_network()
        self._loss_function()

        self.encoder_model.summary()
        self.decoder_model.summary()
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
        xy = concatenate([self.x, self.encoder_labels], axis=1)
        h = Dense(800, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(xy)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(128, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        mean = Dense(self.z_dim, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[self.x, self.encoder_labels], outputs=[mean, log_var, z], name=name)
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
        zy = concatenate([self.z, self.decoder_labels], axis=1)
        h = Dense(128, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(zy)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU(name="mmd")(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
        h = BatchNormalization(axis=1, scale=True)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)

        h = Dense(self.x_dim, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)

        decoder_model = Model(inputs=[self.z, self.decoder_labels], outputs=h, name=name)
        decoder_mmd_model = Model(inputs=[self.z, self.decoder_labels], outputs=self.z, name='decoder_mmd')
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

        reconstruction_output = Lambda(lambda x: x, name="kl_mse")(decoder_output)
        mmd_output = Lambda(lambda x: x, name="mmd")(mmd_output)

        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")

    def _calculate_loss(self):
        loss = LOSSES['mse'](self.mu, self.log_var, self.alpha, self.eta)
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta, self.kernel_method, self.mmd_computation_way)

        return loss, mmd_loss

    def _loss_function(self):
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
        loss, mmd_loss = self._calculate_loss()
        self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=self.clip_value)
        self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
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

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]
        latent = np.nan_to_num(latent)

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def to_mmd_layer(self, adata, encoder_labels):
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
        return self.to_latent(adata, encoder_labels)

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

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        reconstructed = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]
        reconstructed = np.nan_to_num(reconstructed)

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
        self.cvae_model = load_model(os.path.join(self.model_to_use, 'mmd_cvae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self.decoder_mmd_model = load_model(os.path.join(self.model_to_use, 'decoder_mmd.h5'), compile=False)
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_to_use, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
        self.decoder_mmd_model.save(os.path.join(self.model_to_use, "decoder_mmd.h5"), overwrite=True)

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
        train_labels_encoded, _ = label_encoder(train_adata, condition_encoder, condition_key)
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

        if sparse.issparse(train_adata.X):
            train_adata.X = train_adata.X.A

        x = [train_adata.X, train_labels_onehot, train_labels_onehot]
        y = [train_adata.X, train_labels_encoded]

        if valid_adata is not None:
            if sparse.issparse(valid_adata.X):
                valid_adata.X = valid_adata.X.A

            valid_labels_encoded, _ = label_encoder(valid_adata, condition_encoder, condition_key)
            valid_labels_onehot = to_categorical(valid_labels_encoded, num_classes=self.n_conditions)

            x_valid = [valid_adata.X, valid_labels_onehot, valid_labels_onehot]
            y_valid = [valid_adata.X, valid_labels_encoded]

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
