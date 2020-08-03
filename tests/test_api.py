import trvae
import scanpy as sc
import numpy as np

X = np.random.normal(0, 1, size=(1000, 2000))
conditions = list('ABCD') * 250
cell_types = list('abcd') * 250

np.random.shuffle(conditions)
np.random.shuffle(cell_types)

condition_key, cell_type_key = 'condition', 'cell_type'
adata = sc.AnnData(X=X, obs={condition_key: conditions, cell_type_key: cell_types})

model = trvae.models.trVAE(x_dimension=adata.shape[1],
                           conditions=adata.obs[condition_key].unique().tolist(),
                           z_dimension=10,
                           architecture=[128, 32],
                           use_batchnorm=False,
                           loss_fn='mse',
                           eta=50.,
                           alpha=0.0001,
                           beta=50,
                           gene_names=adata.var_names.tolist(),
                           output_activation='linear')

model.train(adata, condition_key,
            train_size=0.8,
            n_epochs=100,
            batch_size=1024,
            early_stop_limit=100,
            lr_reducer=70,
            )
