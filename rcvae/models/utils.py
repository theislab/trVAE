from random import shuffle
import numpy as np
import anndata
from scipy import sparse
from sklearn import preprocessing


def label_encoder(adata, label_encoder=None):
    """
        Encode labels of Annotated `adata` matrix using sklearn.preprocessing.LabelEncoder class.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        Example
        --------
        >>> import rcvae
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = label_encoder(train_data)
    """
    if label_encoder is None:
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(adata.obs["condition"].tolist())
    else:
        le = None
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs['condition'] == condition] = label
    return labels.reshape(-1, 1), le


def shuffle_data(adata, labels=None):
    """
        Shuffles the `adata`.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.
        labels: numpy nd-array
            list of encoded labels

        # Returns
            adata: `~anndata.AnnData`
                Shuffled annotated data matrix.
            labels: numpy nd-array
                Array of shuffled labels if `labels` is not None.

        # Example
        ```python
        import scgen
        import anndata
        import pandas as pd
        train_data = anndata.read("./data/train.h5ad")
        train_labels = pd.read_csv("./data/train_labels.csv", header=None)
        train_data, train_labels = shuffle_data(train_data, train_labels)
        ```
    """
    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    if sparse.issparse(adata.X):
        x = adata.X.A[ind_list, :]
    else:
        x = adata.X[ind_list, :]
    if labels is not None:
        labels = labels[ind_list]
        adata = anndata.AnnData(x, obs={"labels": list(labels)})
        return adata, labels
    else:
        return anndata.AnnData(x, obs=adata.obs)
