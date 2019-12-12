import logging
import os

import anndata
import keras
import numpy as np
from keras.callbacks import CSVLogger, History, EarlyStopping, ReduceLROnPlateau, LambdaCallback, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from trvae.models._activations import ACTIVATIONS
from trvae.models._losses import LOSSES
from trvae.models._utils import sample_z, print_message
from trvae.utils import label_encoder, remove_sparsity

log = logging.getLogger(__file__)


class trVAE:
    """
        trVAE Network class. This class contains the implementation of Regularized Conditional
        Variational Auto-encoder network.
        # Parameters
            kwargs:
                key: `mmd_dimension`: int
                    dimension of MMD layer in trVAE

                key: `kernel_method`: str
                        kernel method for MMD loss calculation

                key: `output_activation`: str
                        activation of output layer in trVAE

                key: `dropout_rate`: float
                        dropout rate

                key: `learning_rate`: float
                    learning rate of optimization algorithm

                key: `model_path`: basestring
                    path to save the model after training

                key: `alpha`: float
                    alpha coefficient for KL loss.

                key: `beta`: float
                    beta coefficient for MMD loss.

                key: `eta`: float
                    eta coefficient for reconstruction loss.

                key: `lambda_l1`: float
                    lambda coefficient for L1 regularization

                key: `lambda_l2`: float
                    lambda coefficient for L2 regularization

                key: `clip_value`: float
                    clip_value for optimizer

            x_dimension: integer
                number of gene expression space dimensions.
            z_dimension: integer
                number of latent space dimensions.
    """

    def __init__(self, x_dimension, n_conditions, z_dimension=20, architecture=None, **kwargs):

        if architecture is None:
            architecture = [256, 128, 32]

        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.mmd_dim = kwargs.get('mmd_dimension', 128)
        self.n_conditions = n_conditions
        self.architecture = architecture

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 1e-5)
        self.beta = kwargs.get("beta", 10)
        self.eta = kwargs.get("eta", 100)
        self.dr_rate = kwargs.get("dropout_rate", 0.1)
        self.model_to_use = kwargs.get("model_path", "./")
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.mmd_calc_mode = kwargs.get("mmd_computation_way", "general")
        self.clip_value = kwargs.get('clip_value', 5)
        self.lambda_l1 = kwargs.get('lambda_l1', 0.0)
        self.lambda_l2 = kwargs.get('lambda_l2', 0.0)

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
                model: Model
                    A keras Model object for Encoder subnetwork of trVAE
        """
        h = concatenate([self.x, self.encoder_labels], axis=1)
        for idx, units in enumerate(self.architecture):
            h = Dense(units, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
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
                decoder_model: Model
                    A keras Model object for Decoder subnetwork of trVAE
                decoder_mmd_model: Model
                    A keras Model object for MMD Decoder subnetwork of trVAE
        """
        h = concatenate([self.z, self.decoder_labels], axis=1)
        for idx, units in enumerate(self.architecture[::-1]):
            h = Dense(units, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=False)(h)
            h = BatchNormalization(axis=1, scale=True)(h)
            if idx == 0:
                h_mmd = LeakyReLU(name="mmd")(h)
                h = h_mmd
            h = Dropout(self.dr_rate)(h)

        h = Dense(self.x_dim, kernel_initializer=self.init_w, kernel_regularizer=self.regularizer, use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)

        decoder_model = Model(inputs=[self.z, self.decoder_labels], outputs=h, name=name)
        decoder_mmd_model = Model(inputs=[self.z, self.decoder_labels], outputs=h_mmd, name='decoder_mmd')
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
        """
            Computes the MSE and MMD loss for trVAE using `LOSSES` dictionary
            # Parameters
                No parameters are needed.
            # Returns
                loss: function
                    mse loss function
                mmd_loss: function
                    mmd loss function
        """
        loss = LOSSES['mse'](self.mu, self.log_var, self.alpha, self.eta)
        mmd_loss = LOSSES['mmd'](self.n_conditions, self.beta, self.kernel_method, self.mmd_calc_mode)

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
        self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
        self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                loss=[loss, mmd_loss],
                                metrics={self.cvae_model.outputs[0].name: loss,
                                         self.cvae_model.outputs[1].name: mmd_loss})

    def to_latent(self, adata, encoder_labels, return_adata=True):
        """
            Map `adata` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.
            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                encoder_labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                return_adata: boolean
                    if `True`, will output as an `anndata` object or put the results in the `obsm` attribute of `adata`
            # Returns
                output: `~anndata.AnnData`
                    returns `anndata` object containing latent space encoding of 'adata'
        """
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]
        latent = np.nan_to_num(latent)

        if return_adata:
            output = anndata.AnnData(X=latent)
            output.obs = adata.obs.copy(deep=True)
        else:
            output = latent

        return output

    def to_mmd_layer(self, adata, encoder_labels, feed_fake=0, return_adata=True):
        """
            Map `adata` in to the MMD layer of trVAE network. This function will compute output
            activation of MMD layer in trVAE.
            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                encoder_labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.
                feed_fake: int
                    if `feed_fake` is non-negative, `decoder_labels` will be identical to `encoder_labels`.
                    if `feed_fake` is not non-negative, `decoder_labels` will be fed with `feed_fake` value.
                return_adata: boolean
                    if `True`, will output as an `anndata` object or put the results in the `obsm` attribute of `adata`
            # Returns
                output: `~anndata.AnnData`
                    returns `anndata` object containing MMD latent space encoding of 'adata'
        """
        if feed_fake >= 0:
            decoder_labels = np.zeros(shape=encoder_labels.shape) + feed_fake
        else:
            decoder_labels = encoder_labels

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        adata = remove_sparsity(adata)

        x = [adata.X, encoder_labels, decoder_labels]
        mmd_latent = self.cvae_model.predict(x)[1]
        mmd_latent = np.nan_to_num(mmd_latent)
        if return_adata:
            output = anndata.AnnData(X=mmd_latent)
            output.obs = adata.obs.copy(deep=True)
        else:
            output = mmd_latent

        return output

    def predict(self, adata, encoder_labels, decoder_labels, return_adata=True):
        """
            Predicts the cell type provided by the user in stimulated condition.
            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.
                encoder_labels: `numpy nd-array`
                    `numpy nd-array` of labels to be fed as encoder's condition array.
                decoder_labels: `numpy nd-array`
                    `numpy nd-array` of labels to be fed as decoder's condition array.
                return_adata: boolean
                    if `True`, will output as an `anndata` object or put the results in the `obsm` attribute of `adata`
            # Returns
                output: `~anndata.AnnData`
                    `anndata` object of predicted cells in primary space.
            # Example
            ```python
            import scanpy as sc
            import trvae
            train_data = sc.read("train.h5ad")
            valid_adata = sc.read("validation.h5ad")
            n_conditions = len(train_adata.obs['condition'].unique().tolist())
            network = trvae.archs.trVAEMulti(train_adata.shape[1], n_conditions)
            network.train(train_adata, valid_adata, le=None,
                          condition_key="condition", cell_type_key="cell_label",
                          n_epochs=1000, batch_size=256
                          )
            encoder_labels, _ = trvae.utils.label_encoder(train_adata, condition_key="condition")
            decoder_labels, _ = trvae.utils.label_encoder(train_adata, condition_key="condition")
            pred_adata = network.predict(train_adata, encoder_labels, decoder_labels)
            ```
        """
        adata = remove_sparsity(adata)

        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_conditions)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_conditions)

        reconstructed = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]
        reconstructed = np.nan_to_num(reconstructed)

        if return_adata:
            output = anndata.AnnData(X=reconstructed)
            output.obs = adata.obs.copy(deep=True)
            output.var_names = adata.var_names
        else:
            output = reconstructed

        return output

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
            import trvae
            train_data = sc.read("train.h5ad")
            valid_adata = sc.read("validation.h5ad")
            n_conditions = len(train_adata.obs['condition'].unique().tolist())
            network = trvae.archs.trVAEMulti(train_adata.shape[1], n_conditions)
            network.restore_model()
            ```
        """
        self.cvae_model = load_model(filepath=os.path.join(self.model_to_use, "best_model.h5"), compile=False)
        self.encoder_model = self.cvae_model.get_layer("encoder")
        self.decoder_model = self.cvae_model.get_layer("decoder")
        self.decoder_mmd_model = self.cvae_model.get_layer("decoder_mmd")
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_to_use, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_to_use, "best_model.h5"), overwrite=True)

    def train(self, train_adata, valid_adata=None,
              condition_encoder=None, condition_key='condition',
              n_epochs=10000, batch_size=1024,
              early_stop_limit=100, lr_reducer=80, threshold=0.0, monitor='val_loss',
              shuffle=True, verbose=0, save=True, monitor_best=True):
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent overfitting.
            # Parameters
                train_adata: `~anndata.AnnData`
                    `AnnData` object for training trVAE
                valid_adata: `~anndata.AnnData`
                    `AnnData` object for validating trVAE (if None, trVAE will automatically split the data with
                    fraction of 80%/20%.
                condition_encoder: dict
                    dictionary of encoded conditions (if None, trVAE will make one for data)
                condition_key: str
                    name of conditions (domains) column in obs matrix
                cell_type_key: str
                    name of cell_types (labels) column in obs matrix
                n_epochs: int
                    number of epochs to iterate and optimize network weights
                batch_size: int
                    number of samples to be used in each batch for network weights optimization
                early_stop_limit: int
                    number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                monitor: str
                    metric to be monitored for early stopping.
                shuffle: boolean
                    if `True`, `train_adata` will be shuffled before training.
                verbose: int
                    level of verbosity
                save: boolean
                    if `True`, the model will be saved in the specified path after training.
            # Returns
                Nothing will be returned
            # Example
            ```python
            import scanpy as sc
            import trvae
            train_data = sc.read("train.h5ad")
            valid_adata = sc.read("validation.h5ad")
            n_conditions = len(train_adata.obs['condition'].unique().tolist())
            network = trvae.archs.trVAEMulti(train_adata.shape[1], n_conditions)
            network.train(train_adata, valid_adata, le=None,
                          condition_key="condition", cell_type_key="cell_label",
                          n_epochs=1000, batch_size=256
                          )
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
        if monitor_best:
            os.makedirs(self.model_to_use, exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=os.path.join(self.model_to_use, "best_model.h5"),
                                             save_best_only=True, monitor=monitor, period=50))

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

            history = self.cvae_model.fit(x=x,
                                          y=y,
                                          epochs=n_epochs,
                                          batch_size=batch_size,
                                          validation_data=(x_valid, y_valid),
                                          shuffle=shuffle,
                                          callbacks=callbacks,
                                          verbose=fit_verbose)
        else:
            history = self.cvae_model.fit(x=x,
                                          y=y,
                                          epochs=n_epochs,
                                          batch_size=batch_size,
                                          validation_split=0.2,
                                          shuffle=shuffle,
                                          callbacks=callbacks,
                                          verbose=fit_verbose)
            if monitor_best:
                self.restore_model()
            elif save and not monitor_best:
                self.save_model()

    def get_corrected(self, adata, labels, return_z=False):
        """
            Computes all trVAE's outputs (Latent (optional), MMD Latent, reconstruction)
            # Parameters
                adata: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.
                labels: `numpy nd-array`
                    `numpy nd-array` of labels to be fed as encoder's condition array.
                return_z: boolean
                    if `True`, will also put z_latent in the `obsm` attribute of `adata`
            # Returns
                Nothing will be returned.
            # Example
            ```python
            import scanpy as sc
            import trvae
            train_data = sc.read("train.h5ad")
            valid_adata = sc.read("validation.h5ad")
            n_conditions = len(train_adata.obs['condition'].unique().tolist())
            network = trvae.archs.trVAEMulti(train_adata.shape[1], n_conditions)
            network.restore_model()
            ```
        """
        reference_labels = np.zeros(adata.shape[0])
        adata.obsm['mmd_latent'] = self.to_mmd_layer(adata, labels, -1, return_adata=False)
        adata.obsm['reconstructed'] = self.predict(adata, reference_labels, labels, return_adata=False)
        if return_z:
            adata.obsm['z_latent'] = self.to_latent(adata, labels, return_adata=False)

    def get_reconstruction_error(self, adata, condition_key):
        adata = remove_sparsity(adata)

        labels_encoded, _ = label_encoder(adata, None, condition_key)
        labels_onehot = to_categorical(labels_encoded, num_classes=self.n_conditions)

        x = [adata.X, labels_onehot, labels_onehot]
        y = [adata.X, labels_encoded]

        return self.cvae_model.evaluate(x, y, verbose=0)
