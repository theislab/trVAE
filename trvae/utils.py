from random import shuffle

import numpy as np
import scanpy as sc
from scipy import sparse


def normalize_hvg(adata, target_sum=1e4, size_factors=True, scale_input=True, logtrans_input=True,
                  n_top_genes=2000):

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_total(adata, target_sum=target_sum, key_added='size_factors')
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if scale_input:
        sc.pp.scale(adata)

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    return adata


def train_test_split(adata, train_frac=0.85):
    """
        Split ``adata`` into train and test annotated datasets.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        train_frac: float
            Fraction of observations (cells) to be used in training dataset. Has to be a value between 0 and 1.

        Returns
        -------
        train_adata: :class:`~anndata.AnnData`
            Training annotated dataset.
        valid_adata: :class:`~anndata.AnnData`
            Test annotated dataset.
    """
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def label_encoder(adata, le=None, condition_key='condition'):
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
        >>> import trvae
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = label_encoder(train_data)
    """
    if le is not None:
        assert isinstance(le, dict)

    unique_conditions = np.unique(adata.obs[condition_key]).tolist()
    if le is None:
        le = dict()
        for idx, condition in enumerate(unique_conditions):
            le[condition] = idx

    assert set(unique_conditions).issubset(list(le.keys()))
    labels = np.zeros(adata.shape[0])
    for condition, label in le.items():
        labels[adata.obs[condition_key] == condition] = label

    return labels.reshape(-1, 1), le


def create_dictionary(conditions, target_conditions=[]):
    if isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary
