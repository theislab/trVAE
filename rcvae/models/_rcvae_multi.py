import logging
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger, History, EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from scipy import sparse

from rcvae.models.layers import SliceLayer, ColwiseMultLayer
from rcvae.models.losses import ZINB, NB
from rcvae.models.utils import label_encoder, shuffle_data

log = logging.getLogger(__file__)


class RCVAEMulti:
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
        self.mmd_dim = kwargs.get('mmd_dimension', 128)
        self.n_conditions = n_conditions

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.conditions = kwargs.get("condition_list")
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.train_with_fake_labels = kwargs.get("train_with_fake_labels", False)
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.arch_style = kwargs.get("arch_style", 1)
        self.use_leaky_relu = kwargs.get("use_leaky_relu", False)
        self.loss_fn = kwargs.get("loss_fn", 'nb')
        self.ridge = kwargs.get('ridge', 0.1)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.encoder_labels = Input(shape=(1,), name="encoder_labels")
        self.decoder_labels = Input(shape=(1,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        if self.loss_fn != "mse":
            self.size_factor = Input(shape=(1,), name='size_factor')

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function()

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()

    def _encoder(self, x, y, name="encoder"):
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
        xy = concatenate([x, y], axis=1)
        if self.arch_style == 1:
            h = Dense(700, kernel_initializer=self.init_w, use_bias=False)(xy)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(400, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
        else:
            h = Dense(32, kernel_initializer=self.init_w, use_bias=False)(xy)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(16, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(self._sample_z, output_shape=(self.z_dim,))([mean, log_var])
        model = Model(inputs=[x, y], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _mmd_decoder(self, z, y, name="decoder"):
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
        zy = concatenate([z, y], axis=1)
        if self.arch_style == 1:
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization()(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dropout(self.dr_rate)(h_mmd)
            h = Dense(400, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(700, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)
            h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)

            if self.loss_fn == 'mse':
                h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
                h = LeakyReLU(name="reconstruction_output")(h)
                if self.use_leaky_relu:
                    h = LeakyReLU(name='reconstruction_output')(h)
                else:
                    h = Activation('relu', name="reconstruction_output")(h)

            elif self.loss_fn == 'nb':
                mean_activation = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
                disp_activation = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
                h_mean = Dense(self.x_dim, activation=mean_activation, kernel_initializer=self.init_w,
                               name='decoder_mean',
                               use_bias=True)(h)
                h_disp = Dense(self.x_dim, activation=disp_activation, kernel_initializer=self.init_w,
                               name='decoder_disp',
                               use_bias=True)(h)
            elif self.loss_fn == 'zinb':
                mean_activation = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
                disp_activation = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
                h_pi = Dense(self.x_dim, activation='sigmoid', kernel_initializer=self.init_w, use_bias=True,
                             name='decoder_pi')(h)
                h_mean = Dense(self.x_dim, activation=mean_activation, kernel_initializer=self.init_w,
                               name='decoder_mean',
                               use_bias=True)(h)
                h_disp = Dense(self.x_dim, activation=disp_activation, kernel_initializer=self.init_w,
                               name='decoder_disp',
                               use_bias=True)(h)
        else:
            h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
            h = BatchNormalization()(h)
            h_mmd = LeakyReLU(name="mmd")(h)
            h = Dense(16, kernel_initializer=self.init_w, use_bias=False)(h_mmd)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            h = Dense(32, kernel_initializer=self.init_w, use_bias=False)(h)
            h = BatchNormalization(axis=1)(h)
            h = LeakyReLU()(h)
            h = Dropout(self.dr_rate)(h)

            h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
            h = LeakyReLU(name="reconstruction_output")(h)
        if self.loss_fn == "mse":
            model = Model(inputs=[z, y], outputs=[h, h_mmd], name=name)
        elif self.loss_fn == 'nb':
            model = Model(inputs=[z, y], outputs=[h_mean, h_mmd, h_disp], name=name)
        elif self.loss_fn == 'zinb':
            model = Model(inputs=[z, y], outputs=[h_mean, h_mmd, h_pi, h_disp], name=name)
        return h, h_mmd, model

    @staticmethod
    def _sample_z(args):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.
            # Parameters
                No parameters are needed.
            # Returns
                The computed Tensor of samples with shape [size, z_dim].
        """
        mu, log_var = args
        batch_size = K.shape(mu)[0]
        z_dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=[batch_size, z_dim])
        return mu + K.exp(log_var / 2) * eps

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
        self.mu, self.log_var, self.encoder_model = self._encoder(*inputs[:2], name="encoder")
        self.x_hat, self.mmd_hl, self.decoder_model = self._mmd_decoder(self.z, self.decoder_labels,
                                                                        name="decoder")
        decoder_outputs = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        if self.loss_fn == 'mse':
            reconstruction_output = Lambda(lambda x: x, name="kl_mse")(decoder_outputs[0])
        elif self.loss_fn == 'nb':
            self.mean_output = Lambda(lambda x: x, name="mean_output")(decoder_outputs[0])
            self.mean_output = ColwiseMultLayer([self.mean_output, self.size_factor])
            self.disp_output = Lambda(lambda x: x, name='disp_output')(decoder_outputs[2])
            reconstruction_output = SliceLayer(0, name='kl_nb')([self.mean_output, self.disp_output])

            inputs += [self.size_factor]
        elif self.loss_fn == 'zinb':
            self.mean_output = Lambda(lambda x: x, name="mean_output")(decoder_outputs[0])
            self.mean_output = ColwiseMultLayer([self.mean_output, self.size_factor])
            self.pi_output = Lambda(lambda x: x, name='pi_output')(decoder_outputs[2])
            self.disp_output = Lambda(lambda x: x, name='disp_output')(decoder_outputs[3])
            reconstruction_output = SliceLayer(0, name='kl_zinb')([self.mean_output, self.pi_output, self.disp_output])

            inputs += [self.size_factor]

        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_outputs[1])
        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")

    @staticmethod
    def compute_kernel(x, y, kernel='rbf', **kwargs):
        """
            Computes RBF kernel between x and y.
            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]
            # Returns
                returns the computed RBF kernel between x and y
        """
        scales = kwargs.get("scales", [])
        if kernel == "rbf":
            x_size = K.shape(x)[0]
            y_size = K.shape(y)[0]
            dim = K.shape(x)[1]
            tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
            tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
            return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
        elif kernel == 'raphy':
            scales = K.variable(value=np.asarray(scales))
            squared_dist = K.expand_dims(RCVAEMulti.squared_distance(x, y), 0)
            scales = K.expand_dims(K.expand_dims(scales, -1), -1)
            weights = K.eval(K.shape(scales)[0])
            weights = K.variable(value=np.asarray(weights))
            weights = K.expand_dims(K.expand_dims(weights, -1), -1)
            return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
        elif kernel == "multi-scale-rbf":
            sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

            beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
            distances = RCVAEMulti.squared_distance(x, y)
            s = K.dot(beta, K.reshape(distances, (1, -1)))

            return K.reshape(tf.reduce_sum(tf.exp(-s), 0), K.shape(distances)) / len(sigmas)

    @staticmethod
    def squared_distance(x, y):  # returns the pairwise euclidean distance
        r = K.expand_dims(x, axis=1)
        return K.sum(K.square(r - y), axis=-1)

    @staticmethod
    def compute_mmd(x, y, kernel, **kwargs):  # [batch_size, z_dim] [batch_size, z_dim]
        """
            Computes Maximum Mean Discrepancy(MMD) between x and y.
            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]
            # Returns
                returns the computed MMD between x and y
        """
        x_kernel = RCVAEMulti.compute_kernel(x, x, kernel=kernel, **kwargs)
        y_kernel = RCVAEMulti.compute_kernel(y, y, kernel=kernel, **kwargs)
        xy_kernel = RCVAEMulti.compute_kernel(x, y, kernel=kernel, **kwargs)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

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

        def batch_loss():
            def zinb_loss(pi, disp, ridge):
                kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)

                def zinb(y_true, y_pred):
                    zinb_obj = ZINB(pi, theta=disp, ridge_lambda=ridge)
                    return zinb_obj.loss(y_true, y_pred) + self.alpha * kl_loss

                return zinb

            def nb_loss(disp):
                kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)

                def nb(y_true, y_pred):
                    nb_obj = NB(theta=disp, masking=False, scale_factor=1.0)
                    return nb_obj.loss(y_true, y_pred) + self.alpha * kl_loss

                return nb

            def kl_recon_loss(y_true, y_pred):
                kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)
                recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
                return recon_loss + self.alpha * kl_loss

            def mmd_loss(real_labels, y_pred):
                with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
                    real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
                    if self.n_conditions == 3:
                        source_mmd, dest1_mmd, dest2_mmd = tf.dynamic_partition(y_pred, real_labels,
                                                                                num_partitions=self.n_conditions)
                        loss = self.compute_mmd(source_mmd, dest1_mmd, self.kernel_method)
                        loss += self.compute_mmd(source_mmd, dest2_mmd, self.kernel_method)
                        loss += self.compute_mmd(dest1_mmd, dest2_mmd, self.kernel_method)
                    elif self.n_conditions == 4:
                        source1_mmd, source2_mmd, source3_mmd, dest_mmd = tf.dynamic_partition(y_pred, real_labels,
                                                                                               num_partitions=self.n_conditions)

                        loss = self.compute_mmd(source1_mmd, source2_mmd, self.kernel_method)
                        loss += self.compute_mmd(source1_mmd, source3_mmd, self.kernel_method)
                        loss += self.compute_mmd(source1_mmd, dest_mmd, self.kernel_method)
                        loss += self.compute_mmd(source2_mmd, source3_mmd, self.kernel_method)
                        loss += self.compute_mmd(source2_mmd, dest_mmd, self.kernel_method)
                        loss += self.compute_mmd(source3_mmd, dest_mmd, self.kernel_method)

                    else:
                        conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=self.n_conditions)
                        loss = 0.0

                        for i in range(len(conditions_mmd)):
                            for j in range(i):
                                loss += self.compute_mmd(conditions_mmd[j], conditions_mmd[j + 1], self.kernel_method)
                    return self.beta * loss

            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            if self.loss_fn == 'mse':
                self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                        loss=[kl_recon_loss, mmd_loss],
                                        metrics={self.cvae_model.outputs[0].name: kl_recon_loss,
                                                 self.cvae_model.outputs[1].name: mmd_loss})
            elif self.loss_fn == 'nb':
                self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                        loss=[nb_loss(self.disp_output), mmd_loss],
                                        metrics={
                                            self.cvae_model.outputs[0].name: nb_loss(self.disp_output),
                                            self.cvae_model.outputs[1].name: mmd_loss})
            elif self.loss_fn == 'zinb':
                self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                        loss=[zinb_loss(self.pi_output, self.disp_output, ridge=self.ridge), mmd_loss],
                                        metrics={
                                            self.cvae_model.outputs[0].name: zinb_loss(self.pi_output, self.disp_output,
                                                                                       ridge=self.ridge),
                                            self.cvae_model.outputs[1].name: mmd_loss})

        batch_loss()

    def to_latent(self, data, labels):
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
        latent = self.encoder_model.predict([data, labels])[2]
        return latent

    def to_mmd_layer(self, model, data, encoder_labels, feed_fake=0):
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
        mmd_latent = model.cvae_model.predict([data, encoder_labels, decoder_labels])[1]
        return mmd_latent

    def _reconstruct(self, data, encoder_labels, decoder_labels, use_data=False):
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
            latent = self.to_latent(data, encoder_labels)
        rec_data = self.decoder_model.predict([latent, decoder_labels])
        return rec_data

    def predict(self, data, encoder_labels, decoder_labels, data_space='None'):
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
                stim_pred = self._reconstruct(data.X.A, encoder_labels, decoder_labels, use_data=True)
            elif data_space == 'mmd':
                stim_pred = self._reconstruct_from_mmd(data.X.A)
            else:
                stim_pred = self._reconstruct(data.X.A, encoder_labels, decoder_labels)
        else:
            if data_space == 'latent':
                stim_pred = self._reconstruct(data.X, encoder_labels, decoder_labels, use_data=True)
            elif data_space == 'mmd':
                stim_pred = self._reconstruct_from_mmd(data.X)
            else:
                stim_pred = self._reconstruct(data.X, encoder_labels, decoder_labels)
        return stim_pred[0]

    def _reconstruct_from_mmd(self, data):
        model = Model(inputs=self.decoder_model.layers[1], outputs=self.decoder_model.outputs)
        return model.predict(data)

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
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_to_use, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_to_use}. Training finished")

    def train(self, train_data, le=None, condition_key='condition', use_validation=False, valid_data=None, n_epochs=25,
              batch_size=32, early_stop_limit=20,
              threshold=0.0025, initial_run=True, monitor='val_loss',
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

        train_labels, _ = label_encoder(train_data, le, condition_key)
        pseudo_labels = np.ones(shape=train_labels.shape)

        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        callbacks = [
            History(),
            EarlyStopping(patience=early_stop_limit, monitor=monitor, min_delta=threshold),
            CSVLogger(filename="./csv_logger.log")
        ]

        if sparse.issparse(train_data.X):
            train_data.X = train_data.X.A
            train_data.raw.X = train_data.raw.X.A

        if shuffle:
            train_data, train_labels = shuffle_data(train_data, train_labels)

        if self.loss_fn != 'mse':
            x = [train_data.X, train_labels, train_labels, train_data.obs['size_factors'].values]
            y = [train_data.raw.X, train_labels]
        else:
            x = [train_data.X, train_labels, train_labels]
            y = [train_data.X, train_labels]

        if use_validation:
            if sparse.issparse(valid_data.X):
                valid_data.X = valid_data.X.A

            valid_labels, _ = label_encoder(valid_data, le, condition_key)

            if shuffle:
                valid_data, valid_labels = shuffle_data(valid_data, valid_labels)

            if self.loss_fn != 'mse':
                x_valid = [valid_data.X, valid_labels, valid_labels, valid_data.obs['size_factors'].values]
                y_valid = [valid_data.raw.X, valid_labels]
            else:
                x_valid = [valid_data.X, valid_labels, valid_labels]
                y_valid = [valid_data.X, valid_labels]

            histories = self.cvae_model.fit(
                x=x,
                y=y,
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        else:
            histories = self.cvae_model.fit(
                x=x,
                y=y,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
        return histories
