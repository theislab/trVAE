import logging
import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from .utils import label_encoder

log = logging.getLogger(__file__)


class RCVAEATAC:
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

    def __init__(self, x_dimension, n_classes, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.n_classes = n_classes
        self.mmd_dim = kwargs.get('mmd_dimension', 128)

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.gamma = kwargs.get("gamma", 1.0)
        self.conditions = kwargs.get("condition_list")
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.train_with_fake_labels = kwargs.get("train_with_fake_labels", False)
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")

        self.x = Input(shape=(self.x_dim,), name="data")
        self.encoder_labels = Input(shape=(1,), name="encoder_labels")
        self.decoder_labels = Input(shape=(1,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")

        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function()

        self.encoder_model.summary()
        self.decoder_model.summary()
        self.cvae_model.summary()
        self.classifier_model.summary()

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
        h = Dense(700, kernel_initializer=self.init_w, use_bias=False)(xy)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(400, kernel_initializer=self.init_w, use_bias=False)(h)
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
        h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(zy)
        h = BatchNormalization()(h)
        h_mmd = LeakyReLU(name="mmd")(h)
        h = Dense(400, kernel_initializer=self.init_w, use_bias=False)(h_mmd)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dense(700, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
        h = LeakyReLU(name="reconstruction_output")(h)

        h_class = Dense(32, kernel_initializer=self.init_w, use_bias=False)(h_mmd)
        h_class = BatchNormalization()(h_class)
        h_class = LeakyReLU()(h_class)

        h_class = Dense(self.n_classes, activation='softmax', name='classifier',
                        kernel_initializer=self.init_w, use_bias=True)(h_class)
        model = Model(inputs=[z, y], outputs=[h, h_mmd], name=name)
        classifier = Model(inputs=[z, y], outputs=[h_class], name='classifier_model')
        return h, h_mmd, model, classifier

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
        self.x_hat, self.mmd_hl, self.decoder_model, self.classifier_model = self._mmd_decoder(self.z,
                                                                                               self.decoder_labels,
                                                                                               name="decoder")
        decoder_outputs = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        classifier_outputs = self.classifier_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_outputs[0])
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_outputs[1])
        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")
        self.classifier_model = Model(inputs=inputs,
                                      outputs=classifier_outputs,
                                      name='classifier')

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
            squared_dist = K.expand_dims(RCVAEATAC.squared_distance(x, y), 0)
            scales = K.expand_dims(K.expand_dims(scales, -1), -1)
            weights = K.eval(K.shape(scales)[0])
            weights = K.variable(value=np.asarray(weights))
            weights = K.expand_dims(K.expand_dims(weights, -1), -1)
            return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
        elif kernel == "multi-scale-rbf":
            sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

            beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
            distances = RCVAEATAC.squared_distance(x, y)
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
        x_kernel = RCVAEATAC.compute_kernel(x, x, kernel=kernel, **kwargs)
        y_kernel = RCVAEATAC.compute_kernel(y, y, kernel=kernel, **kwargs)
        xy_kernel = RCVAEATAC.compute_kernel(x, y, kernel=kernel, **kwargs)
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
            def kl_recon_loss(y_true, y_pred):
                kl_loss = 0.5 * K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, 1)
                recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
                return recon_loss + self.alpha * kl_loss

            def mmd_loss(real_labels, y_pred):
                with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
                    real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
                    source_mmd, dest_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=2)
                    loss = self.compute_mmd(source_mmd, dest_mmd, self.kernel_method)
                    return self.beta * loss

            def cce_loss(real_classes, pred_classes):
                return self.gamma * K.categorical_crossentropy(real_classes, pred_classes)

            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.classifier_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                    loss=[kl_recon_loss, mmd_loss],
                                    metrics={self.cvae_model.outputs[0].name: kl_recon_loss,
                                             self.cvae_model.outputs[1].name: mmd_loss,
                                             })
            self.classifier_model.compile(optimizer=self.classifier_optimizer,
                                          loss=cce_loss,
                                          metrics=['acc'])

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
        self.classifier_model = load_model(os.path.join(self.model_to_use, 'classifier.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_to_use, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
        self.classifier_model.save(os.path.join(self.model_to_use, "classifier.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_to_use}. Training finished")

    def train(self, train_data, condition_key, cell_type_key, source_key, use_validation=False, valid_data=None,
              n_epochs=25, batch_size=32, early_stop_limit=20,
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

        train_labels, _ = label_encoder(train_data)

        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        source_data = train_data.copy()[train_data.obs[condition_key] == source_key]
        source_labels = source_data.obs[cell_type_key].values
        le = LabelEncoder()
        source_labels = le.fit_transform(source_labels)
        source_labels = to_categorical(source_labels, num_classes=self.n_classes)

        n_batches = train_data.shape[0] // batch_size

        def print_message(epoch, cvae_loss, cvae_mmd_loss, cvae_kl_recon_loss,
                          class_cce_loss, class_accuracy, next_is_valid=False, valid=False):
            if not valid:
                if not next_is_valid:
                    message = f"Epoch {epoch}/{n_epochs}:\t[CVAE_loss: {cvae_loss}][KL_Reconstruction_loss: {cvae_kl_recon_loss}][MMD_loss: {cvae_mmd_loss}][CCE_Loss: {class_cce_loss}][CCE_Acc: {class_accuracy}]\r"
                else:
                    message = f"Epoch {epoch}/{n_epochs}:\t[CVAE_loss: {cvae_loss}][KL_Reconstruction_loss: {cvae_kl_recon_loss}][MMD_loss: {cvae_mmd_loss}][CCE_Loss: {class_cce_loss}][CCE_Acc: {class_accuracy}]"
                sys.stdout.write(message)
                sys.stdout.flush()
            else:
                message = f'[CVAE_loss_valid: {cvae_loss_valid}][KL_Reconstruction_loss: {cvae_kl_recon_loss_valid}][MMD_loss: {cvae_mmd_loss_valid}][CCE_Loss: {class_cce_loss_valid}][CCE_Acc: {class_accuracy_valid}]\r'
                sys.stdout.write(message)

        for i in range(n_epochs):
            cvae_loss = 0.0
            cvae_mmd_loss = 0.0
            cvae_kl_recon_loss = 0.0

            class_cce_loss = 0.0
            class_accuracy = 0.0
            for batch_idx in range(n_batches):
                cvae_train_data_batch = train_data.X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                cvae_train_labels_batch = train_labels[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                cvae_loss_batch, cvae_kl_recon_loss_batch, cvae_mmd_loss_batch = self.cvae_model.train_on_batch(
                    x=[cvae_train_data_batch, cvae_train_labels_batch, cvae_train_labels_batch],
                    y=[cvae_train_data_batch, cvae_train_labels_batch],
                )
                class_train_data_batch = source_data.X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                class_train_labels_batch = source_labels[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                class_cce_loss_batch, class_accuracy_batch = self.classifier_model.train_on_batch(
                    x=[class_train_data_batch, np.zeros(class_train_data_batch.shape[0]),
                       np.zeros(class_train_data_batch.shape[0])],
                    y=class_train_labels_batch)

                cvae_loss += cvae_loss_batch / n_batches
                cvae_kl_recon_loss += cvae_kl_recon_loss_batch / n_batches
                cvae_mmd_loss += cvae_mmd_loss_batch / n_batches

                class_cce_loss += class_cce_loss_batch / n_batches
                class_accuracy += class_accuracy_batch / n_batches

                print_message(i, cvae_loss_batch, cvae_mmd_loss_batch, cvae_kl_recon_loss_batch, class_cce_loss_batch,
                              class_accuracy_batch, next_is_valid=False, valid=False)
            # print(f"Epoch {i}/{n_epochs}:\t[CVAE_loss: {cvae_loss}][KL_Reconstruction_loss: {cvae_kl_recon_loss}]"
            #       f"[MMD_loss: {cvae_mmd_loss}][CCE_Loss: {class_cce_loss}][CCE_Acc: {class_accuracy}]", end='')
            print_message(i, cvae_loss, cvae_mmd_loss, cvae_kl_recon_loss, class_cce_loss,
                          class_accuracy, next_is_valid=True, valid=False)
            if use_validation:
                if sparse.issparse(valid_data.X):
                    valid_data.X = valid_data.X.A

                valid_labels, _ = label_encoder(valid_data)

                source_data = valid_data.copy()[valid_data.obs[condition_key] == source_key]
                source_labels = source_data.obs[cell_type_key].values
                le = LabelEncoder()
                source_labels = le.fit_transform(source_labels)
                source_labels = to_categorical(source_labels, num_classes=self.n_classes)

                cvae_loss_valid, cvae_kl_recon_loss_valid, cvae_mmd_loss_valid = self.cvae_model.evaluate(
                    x=[valid_data.X, valid_labels, valid_labels],
                    y=[valid_data.X, valid_labels], verbose=0)

                class_cce_loss_valid, class_accuracy_valid = self.classifier_model.evaluate(
                    x=[source_data.X, np.zeros(source_data.shape[0]), np.zeros(source_data.shape[0], )],
                    y=source_labels, verbose=0)

                # print(f"[CVAE_loss_valid: {cvae_loss_valid}][KL_Reconstruction_loss: {cvae_kl_recon_loss_valid}]"
                #       f"[MMD_loss: {cvae_mmd_loss_valid}][CCE_Loss: {class_cce_loss_valid}][CCE_Acc: "
                #       f"{class_accuracy_valid}]")
                print_message(i, cvae_loss_valid, cvae_mmd_loss_valid, cvae_kl_recon_loss_valid, class_cce_loss_valid,
                              class_accuracy_valid, next_is_valid=False, valid=True)
            else:
                print()

        if save:
            os.makedirs(self.model_to_use, exist_ok=True)
            self.cvae_model.save(os.path.join(self.model_to_use, "mmd_cvae.h5"), overwrite=True)
            self.classifier_model.save(os.path.join(self.model_to_use, "classifier.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
