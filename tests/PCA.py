import os

import anndata
import numpy as np
import scanpy.api as sc
# from data_reader import data_reader
# from hf import *
import scipy.sparse as sparse
from sklearn.decomposition import PCA

# =============================== downloading training and validation files ====================================
# we do not use the validation data to apply vector arithmetics in gene expression space

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")
DATASETS = {
    "Pancreas": {"name": 'pancreas', "source_key": "Baron", "target_key": "Segerstolpe",
                 'train_celltypes': ['alpha', 'beta', 'ductal', 'acinar', 'delta', 'gamma'],
                 'test_celltypes': ['beta'],
                 'cell_type': 'celltype'},

    "PBMC": {"name": 'pbmc', "source_key": "control", "target_key": 'stimulated',
             "cell_type": "cell_type", 'spec_cell_types': ['CD4T', "CD14+Mono", "FCGR3A+Mono"]},

    "Hpoly": {"name": 'hpoly', "source_key": "Control", "target_key": 'Hpoly.Day10',
              "cell_type": "cell_label", 'spec_cell_types': ['Tuft', "Endocrine"]},

    "Salmonella": {"name": 'salmonella', "source_key": "Control", "target_key": 'Salmonella',
                   "cell_type": "cell_label", 'spec_cell_types': ['Tuft', "Endocrine"]},
}


def predict(pca, cd_x, hfd_x, cd_y, p_type="unbiased"):
    if p_type == "unbiased":
        eq = min(len(cd_x), len(hfd_x))
        cd_ind = np.random.choice(range(len(cd_x)), size=eq, replace=False)
        stim_ind = np.random.choice(range(len(hfd_x)), size=eq, replace=False)
    else:
        cd_ind = np.arange(0, len(cd_x))
        stim_ind = np.arange(0, len(hfd_x))
    cd = np.average(cd_x[cd_ind, :], axis=0)
    stim = np.average(hfd_x[stim_ind, :], axis=0)
    delta = stim - cd
    predicted_cells_pca = delta + cd_y
    predicted_cells = pca.inverse_transform(predicted_cells_pca)
    return predicted_cells


def reconstruct():
    train_path = "../data/train_pbmc.h5ad"
    data = sc.read(train_path)
    ctrl_key = "control"
    stim_key = "stimulated"
    all_data = anndata.AnnData()
    print(data.obs["cell_type"].unique().tolist())
    for idx, cell_type in enumerate(data.obs["cell_type"].unique().tolist()):
        pca = PCA(n_components=100)
        train = data[~((data.obs["condition"] == stim_key) & (data.obs["cell_type"] == cell_type))]
        pca.fit(train.X.A)
        print(cell_type, end="\t")
        train_real_stimulated = data[data.obs["condition"] == stim_key, :]
        train_real_stimulated = train_real_stimulated[train_real_stimulated.obs["cell_type"] != cell_type]
        train_real_stimulated = scgen.util.balancer(train_real_stimulated)
        train_real_stimulated_PCA = pca.transform(train_real_stimulated.X)

        train_real_cd = data[data.obs["condition"] == ctrl_key, :]
        train_real_cd = scgen.util.balancer(train_real_cd)
        train_real_cd_PCA = pca.transform(train_real_cd.X)

        cell_type_adata = data[data.obs["cell_type"] == cell_type]
        cell_type_ctrl = cell_type_adata[cell_type_adata.obs["condition"] == ctrl_key]
        cell_type_stim = cell_type_adata[cell_type_adata.obs["condition"] == stim_key]
        if sparse.issparse(cell_type_ctrl.X):
            cell_type_ctrl_PCA = pca.transform(cell_type_ctrl.X.A)
        else:
            cell_type_ctrl_PCA = pca.transform(cell_type_ctrl.X)
        predicted_cells = predict(pca, train_real_cd_PCA, train_real_stimulated_PCA, cell_type_ctrl_PCA)
        if sparse.issparse(cell_type_ctrl.X):
            all_Data = sc.AnnData(np.concatenate([cell_type_ctrl.X.A, cell_type_stim.X.A, predicted_cells]))
        else:
            all_Data = sc.AnnData(np.concatenate([cell_type_ctrl.X, cell_type_stim.X, predicted_cells]))
        all_Data.obs["condition"] = [f"{cell_type}_ctrl"] * cell_type_ctrl.shape[0] + [f"{cell_type}_real_stim"] * \
                                    cell_type_stim.shape[0] + \
                                    [f"{cell_type}_pred_stim"] * len(predicted_cells)
        all_Data.obs["cell_type"] = [f"{cell_type}"] * (
                cell_type_ctrl.shape[0] + cell_type_stim.shape[0] + len(predicted_cells))
        all_Data.var_names = cell_type_adata.var_names

        if idx == 0:
            all_data = all_Data
        else:
            all_data = all_data.concatenate(all_Data)
        print(cell_type)
    sc.write("../data/reconstructed/PCAVecArithm/PCA_pbmc.h5ad", all_data)


def train(data_name="pbmc", cell_type="CD4T", p_type="unbiased"):
    train_path = f"../data/train_{data_name}.h5ad"
    if data_name == "pbmc":
        ctrl_key = "control"
        stim_key = "stimulated"
        cell_type_key = "cell_type"
    elif data_name == "hpoly":
        ctrl_key = "Control"
        stim_key = "Hpoly.Day10"
        cell_type_key = "cell_label"
    elif data_name == "salmonella":
        ctrl_key = "Control"
        stim_key = "Salmonella"
        cell_type_key = "cell_label"
    data = sc.read(train_path)
    print("data has been loaded!")
    train = data[~((data.obs["condition"] == stim_key) & (data.obs[cell_type_key] == cell_type))]
    pca = PCA(n_components=100)

    pca.fit(train.X.A)

    train_real_cd = train[train.obs["condition"] == "control", :]
    if p_type == "unbiased":
        train_real_cd = scgen.util.balancer(train_real_cd)
    train_real_stimulated = train[train.obs["condition"] == "stimulated", :]
    if p_type == "unbiased":
        train_real_stimulated = scgen.util.balancer(train_real_stimulated)

    import scipy.sparse as sparse
    if sparse.issparse(train_real_cd.X):
        train_real_cd.X = train_real_cd.X.A
        train_real_stimulated.X = train_real_stimulated.X.A

    train_real_stimulated_PCA = pca.transform(train_real_stimulated.X)
    train_real_cd_PCA = pca.transform(train_real_cd.X)

    adata_list = scgen.util.extractor(data, cell_type, {"ctrl": ctrl_key, "stim": stim_key})
    if sparse.issparse(adata_list[1].X):
        adata_list[1].X = adata_list[1].X.A
        adata_list[2].X = adata_list[2].X.A
    ctrl_CD4T_PCA = pca.transform(adata_list[1].X)
    predicted_cells = predict(pca, train_real_cd_PCA, train_real_stimulated_PCA, ctrl_CD4T_PCA, p_type)

    all_Data = sc.AnnData(np.concatenate([adata_list[1].X, adata_list[2].X, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * len(adata_list[1].X) + ["real_stim"] * len(adata_list[2].X) + \
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = adata_list[3].var_names
    if p_type == "unbiased":
        sc.write(f"../data/reconstructed/PCAVecArithm/PCA_CD4T.h5ad", all_Data)
    else:
        sc.write(f"../data/reconstructed/PCAVecArithm/PCA_CD4T_biased.h5ad", all_Data)


if __name__ == "__main__":
    # sc.pp.neighbors(all_Data)
    # sc.tl.umap(all_Data)
    # import matplotlib
    # import matplotlib.style
    # import matplotlib.pyplot as plt
    #
    # plt.style.use('default')
    # sc.pl.umap(all_Data, color=["condition"], frameon=False, palette=matplotlib.rcParams["axes.prop_cycle"]
    #            , save="Vec_Arith_PCA_biased.png", show=False,
    #            legend_fontsize=18, title="")
    # sc.pl.violin(all_Data, groupby='condition', keys="ISG15", save="Vec_Arith_PCA.pdf", show=False)
    train("pbmc", "CD4T", "unbiased")
    train("pbmc", "CD4T", "biased")
    reconstruct()
