import logging
import os

import anndata
import keras
import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.utils import to_categorical
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from trvae.models._losses import LOSSES
from trvae.models._activations import ACTIVATIONS
from trvae.utils import remove_sparsity, label_encoder
from ._utils import sample_z

log = logging.getLogger(__file__)


class trVAEATAC:
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

    def __init__(self, x_dimension, n_labels, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.n_labels = n_labels
        self.n_domains = kwargs.get("n_domains", 2)
        self.mmd_dim = kwargs.get('mmd_dimension', 128)

        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.001)
        self.beta = kwargs.get("beta", 100)
        self.prev_beta = kwargs.get("beta", 100)
        self.gamma = kwargs.get("gamma", 1.0)
        self.eta = kwargs.get("eta", 1.0)
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_path = kwargs.get("model_path", "./")
        self.kernel_method = kwargs.get("kernel", "multi-scale-rbf")
        self.output_activation = kwargs.get("output_activation", 'relu')
        self.mmd_computation_way = kwargs.get("mmd_computation_way", "general")
        self.print_summary = kwargs.get("print_summary", True)

        self.x = Input(shape=(self.x_dim,), name="data")
        self.encoder_labels = Input(shape=(self.n_domains,), name="encoder_labels")
        self.decoder_labels = Input(shape=(self.n_domains,), name="decoder_labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")
        self.mmd_latent = Input(shape=(self.mmd_dim,), name="MMD_Latent")

        self.label_enc = LabelEncoder()
        self.domain_enc = LabelEncoder()
        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function()

        if self.print_summary:
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
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(xy)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(self.mmd_dim, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(sample_z, output_shape=(self.z_dim,))([mean, log_var])
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
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h_mmd)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
        h = ACTIVATIONS[self.output_activation](h)
        decoder_model = Model(inputs=[z, y], outputs=h, name=name)
        mmd_model = Model(inputs=[z, y], outputs=h_mmd, name='decoder_mmd')

        return decoder_model, mmd_model

    def _classifier(self, name='classifier_from_mmd_latent'):
        mmd_latent = Input(shape=(self.mmd_dim,))

        h = Dense(self.n_labels, activation='softmax', name='classifier_prob',
                  kernel_initializer=self.init_w, use_bias=True)(mmd_latent)

        model = Model(inputs=mmd_latent, outputs=h, name=name)
        return model

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
        self.decoder_model, self.decoder_mmd_model = self._mmd_decoder(self.z,
                                                                       self.decoder_labels,
                                                                       name="decoder")
        self.classifier_from_mmd_latent_model = self._classifier()

        decoder_output = self.decoder_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])
        decoder_mmd_output = self.decoder_mmd_model([self.encoder_model(inputs[:2])[2], self.decoder_labels])

        reconstruction_output = Lambda(lambda x: x, name="kl_reconstruction")(decoder_output)
        mmd_output = Lambda(lambda x: x, name="mmd")(decoder_mmd_output)

        classifier_output = self.classifier_from_mmd_latent_model(decoder_mmd_output)

        self.cvae_model = Model(inputs=inputs,
                                outputs=[reconstruction_output, mmd_output],
                                name="cvae")
        self.classifier_model = Model(inputs=inputs,
                                      outputs=classifier_output,
                                      name='classifier')

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
            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.classifier_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                    loss=[LOSSES['mse'](self.mu, self.log_var, self.alpha, self.eta),
                                          LOSSES['mmd'](self.n_domains, self.beta, self.kernel_method,
                                                        self.mmd_computation_way)],
                                    metrics={self.cvae_model.outputs[0].name: LOSSES['mse'](self.mu, self.log_var,
                                                                                            self.alpha, self.eta),
                                             self.cvae_model.outputs[1].name: LOSSES['mmd'](self.n_domains, self.beta,
                                                                                            self.kernel_method,
                                                                                            self.mmd_computation_way),
                                             })
            self.classifier_model.compile(optimizer=self.classifier_optimizer,
                                          loss=LOSSES['cce'](self.gamma),
                                          metrics=['acc'])

        batch_loss()

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
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_domains)
        latent = self.encoder_model.predict([adata.X, encoder_labels])[2]

        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)
        latent_adata.var_names = adata.var_names

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
        encoder_labels = to_categorical(encoder_labels, num_classes=self.n_domains)
        decoder_labels = to_categorical(decoder_labels, num_classes=self.n_domains)

        adata = remove_sparsity(adata)
        mmd_latent = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[1]
        mmd_latent_adata = anndata.AnnData(X=mmd_latent)
        mmd_latent_adata.obs = adata.obs.copy(deep=True)

        return mmd_latent_adata

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
        reconstructed = self.cvae_model.predict([adata.X, encoder_labels, decoder_labels])[0]
        reconstructed_adata = anndata.AnnData(X=reconstructed)
        reconstructed_adata.obs = adata.obs.copy(deep=True)
        reconstructed_adata.var_names = adata.var_names

        return reconstructed_adata

    def _make_classifier_trainable(self, trainable):
        self.classifier_model.layers[-1].trainable = trainable

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
        self.classifier_model = load_model(os.path.join(self.model_path, 'classifier.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_path, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_path, 'decoder.h5'), compile=False)
        self.decoder_mmd_model = load_model(os.path.join(self.model_path, 'decoder_mmd.h5'), compile=False)
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        self.cvae_model.save(os.path.join(self.model_path, "mmd_cvae.h5"), overwrite=True)
        self.classifier_model.save(os.path.join(self.model_path, "classifier.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_path, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_path, "decoder.h5"), overwrite=True)
        self.decoder_mmd_model.save(os.path.join(self.model_path, "decoder_mmd.h5"), overwrite=True)

    def train(self, train_adata, valid_adata,
              domain_key, label_key, source_key, target_key, domain_encoder,
              n_epochs=25, batch_size=32, early_stop_limit=20,
              lr_reducer=10, save=True):
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
            train_data = sc.read(train_kang.h5ad)
            validation_data = sc.read(valid_kang.h5ad)
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            ```
        """
        if not isinstance(source_key, list):
            source_key = [source_key]
        if not isinstance(target_key, list):
            target_key = [target_key]

        train_labels_encoded, self.domain_enc = label_encoder(train_adata, label_encoder=domain_encoder, condition_key=domain_key)
        valid_labels_encoded, _ = label_encoder(valid_adata, label_encoder=domain_encoder, condition_key=domain_key)

        train_labels_onehot = to_categorical(train_labels_encoded, num_classes=self.n_domains)
        valid_labels_onehot = to_categorical(valid_labels_encoded, num_classes=self.n_domains)

        train_adata = remove_sparsity(train_adata)
        valid_adata = remove_sparsity(valid_adata)

        source_adata_train = train_adata.copy()[train_adata.obs[domain_key].isin(source_key)]
        source_classes_train = source_adata_train.obs[label_key].values
        source_classes_train = self.label_enc.fit_transform(source_classes_train)
        source_classes_train = to_categorical(source_classes_train, num_classes=self.n_labels)
        source_domains_train_encoded = np.zeros(source_adata_train.shape[0])
        source_domains_train_onehot = to_categorical(source_domains_train_encoded, num_classes=self.n_domains)

        target_adata_train = train_adata.copy()[train_adata.obs[domain_key].isin(target_key)]
        target_classes_train = target_adata_train.obs[label_key].values
        target_classes_train = self.label_enc.transform(target_classes_train)
        target_classes_train = to_categorical(target_classes_train, num_classes=self.n_labels)
        target_domains_encoded = np.ones(target_adata_train.shape[0])
        target_domains_onehot = to_categorical(target_domains_encoded, num_classes=self.n_domains)

        source_adata_valid = valid_adata.copy()[valid_adata.obs[domain_key].isin(source_key)]
        source_classes_valid = source_adata_valid.obs[label_key].values
        source_classes_valid = self.label_enc.transform(source_classes_valid)
        source_classes_valid = to_categorical(source_classes_valid, num_classes=self.n_labels)
        source_domains_valid_encoded = np.zeros(source_adata_valid.shape[0])
        source_domains_valid_onehot = to_categorical(source_domains_valid_encoded, num_classes=self.n_domains)

        best_val_loss = 100000.0
        patience = 0
        for i in range(n_epochs):
            x_train = [train_adata.X, train_labels_onehot, train_labels_onehot]
            y_train = [train_adata.X, train_labels_encoded]

            x_valid = [valid_adata.X, valid_labels_onehot, valid_labels_onehot]
            y_valid = [valid_adata.X, valid_labels_encoded]

            cvae_history = self.cvae_model.fit(
                x=x_train,
                y=y_train,
                epochs=1,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                verbose=0,
            )

            x_train = [source_adata_train.X,
                       source_domains_train_onehot,
                       source_domains_train_onehot]
            y_train = source_classes_train

            x_valid = [source_adata_valid.X,
                       source_domains_valid_onehot,
                       source_domains_valid_onehot]
            y_valid = source_classes_valid

            class_history = self.classifier_model.fit(
                x=x_train,
                y=y_train,
                epochs=1,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                verbose=0,
            )

            x_target = [target_adata_train.X,
                        target_domains_onehot,
                        target_domains_onehot]
            y_target = target_classes_train

            _, target_acc = self.classifier_model.evaluate(x_target, y_target, verbose=0)

            cvae_loss = cvae_history.history['loss'][0]
            cvae_kl_recon_loss = cvae_history.history['kl_reconstruction_loss'][0]
            cvae_mmd_loss = cvae_history.history['mmd_loss'][0]

            cvae_loss_valid = cvae_history.history['val_loss'][0]
            cvae_kl_recon_loss_valid = cvae_history.history['val_kl_reconstruction_loss'][0]
            cvae_mmd_loss_valid = cvae_history.history['val_mmd_loss'][0]

            class_cce_loss = class_history.history['loss'][0]
            class_accuracy = class_history.history['acc'][0]

            class_cce_loss_valid = class_history.history['val_loss'][0]
            class_accuracy_valid = class_history.history['val_acc'][0]
            print(f"Epoch {i + 1}/{n_epochs}:")
            print(f'loss: {cvae_loss:.4f} - KL_Recon_loss: {cvae_kl_recon_loss:.4f}'
                  f" - MMD_loss: {cvae_mmd_loss:.4f} - CCE_Loss: {class_cce_loss:.4f} - CCE_Acc: {class_accuracy:.3f}"
                  f" - val_loss: {cvae_loss_valid:.4f} - val_KL_Recon_loss: {cvae_kl_recon_loss_valid:.4f}"
                  f" - val_MMD_loss: {cvae_mmd_loss_valid:.4f} - val_CCE_Loss: {class_cce_loss_valid:.4f}"
                  f" - val_CCE_Acc: {class_accuracy_valid:.3f} - Target_acc: {target_acc:.3f}")

            if class_cce_loss_valid > best_val_loss:
                patience += 1
                if patience > early_stop_limit:
                    break
            else:
                best_val_loss = class_cce_loss_valid
                patience = 0

            if lr_reducer > 0 and (i + 1) % lr_reducer == 0:
                self.lr = self.lr / 10
                self._loss_function()
                print(f"Epoch {i}: learning rate reduced to {self.lr}")
        if save:
            self.save_model()
