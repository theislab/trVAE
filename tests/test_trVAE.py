import argparse
import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import trvae

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")

from matplotlib import pyplot as plt

DATASETS = {
    "Pancreas": {"name": 'pancreas', "source_key": "Baron", "target_key": "Segerstolpe",
                 'train_celltypes': ['alpha', 'beta', 'ductal', 'acinar', 'delta', 'gamma'],
                 'test_celltypes': ['beta'],
                 'cell_type': 'celltype'},

    "PBMC": {"name": 'pbmc', "source_key": "control", "target_key": 'stimulated',
             "cell_type": "cell_type", 'spec_cell_types': []},

    "Hpoly": {"name": 'hpoly', "source_key": "Control", "target_key": 'Hpoly.Day10',
              "cell_type": "cell_label", 'spec_cell_types': []},

    "Salmonella": {"name": 'salmonella', "source_key": "Control", "target_key": 'Salmonella',
                   "cell_type": "cell_label", 'spec_cell_types': []},

    "multimodal": {"name": 'multimodal', "source_key": "RNA-seq", "target_key": 'ATAC-seq',
                   "cell_type": "modality", 'spec_cell_types': []},

}


def train_network(data_dict=None,
                  z_dim=100,
                  mmd_dimension=256,
                  alpha=0.001,
                  beta=100,
                  kernel='multi-scale-rbf',
                  n_epochs=500,
                  batch_size=512,
                  early_stop_limit=50,
                  dropout_rate=0.2,
                  learning_rate=0.001,
                  ):
    data_name = data_dict['name']
    target_key = data_dict.get('target_key', None)
    cell_type_key = data_dict.get("cell_type", None)

    train_data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"../data/{data_name}/valid_{data_name}.h5ad")
    cell_types = train_data.obs[cell_type_key].unique().tolist()

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type:
        cell_types = spec_cell_type

    for cell_type in cell_types:
        net_train_data = train_data.copy()[
            ~((train_data.obs[cell_type_key] == cell_type) & (train_data.obs['condition'] == target_key))]
        net_valid_data = valid_data.copy()[
            ~((valid_data.obs[cell_type_key] == cell_type) & (valid_data.obs['condition'] == target_key))]

        network = trvae.trVAE(x_dimension=net_train_data.shape[1],
                              z_dimension=z_dim,
                              mmd_dimension=mmd_dimension,
                              alpha=alpha,
                              beta=beta,
                              kernel=kernel,
                              learning_rate=learning_rate,
                              train_with_fake_labels=False,
                              model_path=f"../models/RCVAE/{data_name}/{cell_type}/{z_dim}/",
                              dropout_rate=dropout_rate)

        network.train(net_train_data,
                      use_validation=True,
                      valid_adata=net_valid_data,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      verbose=2,
                      early_stop_limit=early_stop_limit,
                      shuffle=True,
                      save=True)

        print(f"Model for {cell_type} has been trained")


def train_network_multi(data_dict=None,
                        z_dim=100,
                        mmd_dimension=256,
                        alpha=0.001,
                        beta=100,
                        kernel='multi-scale-rbf',
                        n_epochs=500,
                        batch_size=512,
                        early_stop_limit=50,
                        dropout_rate=0.2,
                        learning_rate=0.001,
                        ):
    data_name = data_dict['name']
    target_key = data_dict.get('target_key', None)
    print(data_name)

    train_data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    valid_data = sc.read(f"../data/{data_name}/valid_{data_name}.h5ad")

    network = trvae.trVAE(x_dimension=train_data.shape[1],
                          z_dimension=z_dim,
                          mmd_dimension=mmd_dimension,
                          alpha=alpha,
                          beta=beta,
                          kernel=kernel,
                          learning_rate=learning_rate,
                          train_with_fake_labels=False,
                          model_path=f"../models/RCVAE/{data_name}/{z_dim}/",
                          dropout_rate=dropout_rate)

    network.train(train_data,
                  use_validation=True,
                  valid_adata=valid_data,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=2,
                  early_stop_limit=early_stop_limit,
                  shuffle=False,
                  save=True)


def score(adata, n_deg=10, n_genes=1000, condition_key="condition", cell_type_key="cell_type",
          conditions={"stim": "stimulated", "ctrl": "control"},
          sortby="median_score"):
    import scanpy as sc
    import numpy as np
    from scipy.stats import entropy
    import pandas as pd
    sc.tl.rank_genes_groups(adata, groupby=condition_key, method="wilcoxon", n_genes=n_genes)
    gene_names = adata.uns["rank_genes_groups"]['names'][conditions['stim']]
    gene_lfcs = adata.uns["rank_genes_groups"]['logfoldchanges'][conditions['stim']]
    diff_genes_df = pd.DataFrame({"names": gene_names, "lfc": gene_lfcs})
    diff_genes = diff_genes_df["names"].tolist()[:n_genes]
    print(len(diff_genes))

    adata_deg = adata[:, diff_genes].copy()
    cell_types = adata_deg.obs[cell_type_key].cat.categories.tolist()
    lfc_temp = np.zeros((len(cell_types), n_genes))
    for j, ct in enumerate(cell_types):
        if cell_type_key == "cell_type":  # if data is pbmc
            stim = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0).A1
            ctrl = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0).A1
        else:
            stim = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0)
            ctrl = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0)
        lfc_temp[j] = np.abs((stim - ctrl)[None, :])
    norm_lfc = lfc_temp / lfc_temp.sum(0).reshape((1, n_genes))
    ent_scores = entropy(norm_lfc)
    median = np.median(lfc_temp, axis=0)
    med_scores = np.max(np.abs((lfc_temp - median)), axis=0)
    df_score = pd.DataFrame({"genes": adata_deg.var_names.tolist(), "median_score": med_scores,
                             "entropy_score": ent_scores})
    if sortby == "median_score":
        return df_score.sort_values(by=['median_score'], ascending=False).iloc[:n_deg, :]
    else:
        return df_score.sort_values(by=['entropy_score'], ascending=False).iloc[:n_deg, :]


def plot_boxplot(data_dict,
                 method,
                 n_genes=100,
                 restore=True,
                 score_type="median_score",
                 y_measure="SE",
                 scale="log"):
    data_name = data_dict.get('name', None)
    ctrl_key = data_dict.get("source_key", None)
    stim_key = data_dict.get("target_key", None)
    cell_type_key = data_dict.get("cell_type", None)

    train = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    recon_data = sc.read(f"../data/reconstructed/RCVAE/{data_name}.h5ad")

    import matplotlib
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rc('xtick', labelsize=14)
    conditions = {"ctrl": ctrl_key, "stim": stim_key}

    path_to_save = f"../results/RCVAE/Benchmark/{data_name}/"
    os.makedirs(path_to_save, exist_ok=True)

    sc.settings.figdir = path_to_save

    diff_genes = score(train, n_deg=10 * n_genes, n_genes=500, cell_type_key=cell_type_key,
                       conditions=conditions,
                       sortby=score_type)
    diff_genes = diff_genes["genes"].tolist()

    # epsilon = 1e-7
    os.makedirs(os.path.join(path_to_save, f"./boxplots/Top_{10 * n_genes}/{y_measure}/"), exist_ok=True)
    if not restore:
        n_cell_types = len(train.obs[cell_type_key].unique().tolist())
        all_scores = np.zeros(shape=(n_cell_types * 10 * n_genes, 1))
        for bin_idx in range(10):
            for cell_type_idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
                real_stim = recon_data[(recon_data.obs[cell_type_key] == cell_type) & (
                        recon_data.obs["condition"] == f"{cell_type}_real_stim")]
                pred_stim = recon_data[(recon_data.obs[cell_type_key] == cell_type) & (
                        recon_data.obs["condition"] == f"{cell_type}_pred_stim")]

                real_stim = real_stim[:, diff_genes[bin_idx * n_genes:(bin_idx + 1) * n_genes]]
                pred_stim = pred_stim[:, diff_genes[bin_idx * n_genes:(bin_idx + 1) * n_genes]]
                if sparse.issparse(real_stim.X):
                    real_stim_avg = np.average(real_stim.X.A, axis=0)
                    pred_stim_avg = np.average(pred_stim.X.A, axis=0)
                else:
                    real_stim_avg = np.average(real_stim.X, axis=0)
                    pred_stim_avg = np.average(pred_stim.X, axis=0)
                if y_measure == "SE":  # (x - xhat) ^ 2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE":  # x - xhat
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE:x":  # (x - xhat) / x
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, real_stim_avg)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "SE:x^2":  # (x - xhat) / x^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(real_stim_avg, 2))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE:max(x, 1)":  # (x - xhat) / max(x, 1)
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, np.maximum(real_stim_avg, 1.0))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "SE:max(x, 1)^2":  # (x - xhat)^2 / max(x, 1)^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(np.maximum(real_stim_avg, 1.0), 2))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "SE:max(x, 1)":  # (x - xhat)^2 / max(x, 1)
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(np.maximum(real_stim_avg, 1.0), 1.0))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - AE:x":  # 1 - ((x - xhat) / x)
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, real_stim_avg)
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - SE:x^2":  # 1 - ((x - xhat) / x)^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(real_stim_avg, 2))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - AE:max(x, 1)":  # 1 - ((x - xhat) / max(x, 1.0))
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.true_divide(y_measures, np.maximum(real_stim_avg, 1.0))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - SE:max(x, 1)^2":  # 1 - ((x - xhat) / max(x, 1.0))
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.true_divide(y_measures, np.power(np.maximum(real_stim_avg, 1.0), 2))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                if scale == "log":
                    y_measures_reshaped = np.log(y_measures_reshaped)
                start = n_cell_types * n_genes * bin_idx
                all_scores[start + n_genes * cell_type_idx:start + n_genes * (cell_type_idx + 1),
                0] = y_measures_reshaped
        all_scores = np.reshape(all_scores, (-1,))
        print(all_scores.shape)
    else:
        all_scores = np.loadtxt(
            fname=f"./boxplots/Top_{10 * n_genes}/{y_measure}/y_measures_{score_type}_{n_genes}_({y_measure}).txt",
            delimiter=",")
    import seaborn as sns
    conditions = [f"Bin-{i // (n_cell_types * n_genes) + 1}" for i in range(n_cell_types * 10 * n_genes)]
    all_scores_df = pd.DataFrame({"scores": all_scores})
    all_scores_df["conditions"] = conditions
    ax = sns.boxplot(data=all_scores_df, x="conditions", y="scores", whis=np.inf)
    # if scale != "log" and y_measure == "AE:max(x, 1)":
    #     ax.set_ylim(0.0, 1.75)
    # elif scale != "log" and y_measure == "SE:max(x, 1)":
    #     ax.set_ylim(0.0, 3.0)
    # elif y_measure == "AE:max(x, 1)":
    #     ax.set_ylim(-15.0, 0.5)
    # elif y_measure == "SE:max(x, 1)":
    #     ax.set_ylim(-30.5, 1.0)
    xlabels = ['Bin-%i' % i for i in range(10)]
    ax.set_xticklabels(xlabels, rotation=90)
    if y_measure == "SE":
        plt.ylabel("(x - xhat) ^ 2")
    elif y_measure == "AE":
        plt.ylabel("|x - xhat|")
    elif y_measure == "AE:x":
        plt.ylabel("|x - xhat| / x")
    elif y_measure == "SE:x^2":
        plt.ylabel("((x - xhat) ^ 2) / (x ^ 2)")
    elif y_measure == "AE:max(x, 1)":
        if scale == "log":
            plt.ylabel("log(|x - xhat| / max(x, 1))")
        else:
            plt.ylabel("|x - xhat| / max(x, 1)")
    elif y_measure == "SE:max(x, 1)^2":
        plt.ylabel("(x - xhat)^2 / max(x, 1)^2")
    elif y_measure == "SE:max(x, 1)":
        if scale == "log":
            plt.ylabel("log((x - xhat)^2 / max(x, 1))")
        else:
            plt.ylabel("(x - xhat)^2 / max(x, 1)")
    elif y_measure == "1 - AE:x":
        plt.ylabel("1 - (|x - xhat| / x)")
    elif y_measure == "1 - SE:x^2":
        plt.ylabel("1 - ((x - xhat)^2 / x^2)")
    elif y_measure == "1 - AE:max(x, 1)":
        plt.ylabel("1 - (|x - xhat| / max(x, 1))")
    elif y_measure == "1 - SE:max(x, 1)^2":
        plt.ylabel("1 - ((x - xhat)^2 / max(x, 1)^2)")
    os.makedirs(os.path.join(path_to_save, f"./boxplots/Top_{10 * n_genes}/{y_measure}/"), exist_ok=True)
    plt.tight_layout()
    name = os.path.join(path_to_save,
                        f"./boxplots/Top_{10 * n_genes}/{y_measure}/{method}_{data_name}_boxplot_{score_type}_{n_genes}_{scale}.pdf")
    plt.savefig(name, dpi=300)
    plt.close()


def reconstruct_whole_data(data_dict={}, z_dim=100):
    data_name = data_dict.get('name', None)
    ctrl_key = data_dict.get("source_key", None)
    stim_key = data_dict.get("target_key", None)
    cell_type_key = data_dict.get("cell_type", None)
    train = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")

    if sparse.issparse(train.X):
        train.X = train.X.A

    all_data = anndata.AnnData()
    cell_types = train.obs[cell_type_key].unique().tolist()

    for idx, cell_type in enumerate(cell_types):
        print(f"Reconstructing for {cell_type}")
        network = trvae.trVAE(x_dimension=train.shape[1],
                              z_dimension=z_dim,
                              model_path=f"../models/RCVAE/{data_name}/{cell_type}/{z_dim}/",
                              )
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs["condition"] == ctrl_key))]
        pred = network.predict(cell_type_ctrl_data,
                               encoder_labels=np.zeros((cell_type_ctrl_data.shape[0], 1)),
                               decoder_labels=np.ones((cell_type_ctrl_data.shape[0], 1)))

        pred_adata = anndata.AnnData(pred,
                                     obs={"condition": [f"{cell_type}_pred_stim"] * len(pred),
                                          cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})

        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={"condition": [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})

        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs["condition"] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs["condition"] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={"condition": [f"{cell_type}_real_stim"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        print(f"Finish Reconstructing for {cell_type}")
    all_data.write_h5ad(f"../data/reconstructed/RCVAE/{data_name}.h5ad")


def stacked_violin_plot(data_dict, method, score_type="median_score"):
    data_name = data_dict.get('name', None)
    ctrl_key = data_dict.get("source_key", None)
    stim_key = data_dict.get("target_key", None)
    cell_type_key = data_dict.get("cell_type", None)

    train = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    recon_data = sc.read(f"../data/reconstructed/RCVAE/{data_name}.h5ad")

    path_to_save = f"../results/RCVAE/Benchmark/{data_name}/"
    sc.settings.figdir = path_to_save
    conditions = {"ctrl": ctrl_key, "stim": stim_key}

    diff_genes = score(train, n_deg=10, n_genes=500, cell_type_key=cell_type_key, conditions=conditions,
                       sortby=score_type)
    diff_genes = diff_genes["genes"].tolist()
    sc.pl.stacked_violin(recon_data,
                         var_names=diff_genes,
                         groupby="condition",
                         save=f"_{method}_{score_type}_{data_name}.pdf",
                         swap_axes=True,
                         show=True)


def visualize_trained_network_results(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)
    cell_type_key = data_dict.get("cell_type", None)

    data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    cell_types = data.obs[cell_type_key].unique().tolist()

    spec_cell_type = data_dict.get("spec_cell_types", None)
    if spec_cell_type:
        cell_types = spec_cell_type

    for cell_type in cell_types:
        path_to_save = f"../results/RCVAE/{data_name}/{cell_type}/{z_dim}/{source_key} to {target_key}/Visualizations/"
        os.makedirs(path_to_save, exist_ok=True)
        sc.settings.figdir = os.path.abspath(path_to_save)

        train_data = data.copy()[~((data.obs['condition'] == target_key) & (data.obs[cell_type_key] == cell_type))]

        cell_type_adata = data[data.obs[cell_type_key] == cell_type]

        network = trvae.trVAE(x_dimension=data.shape[1],
                              z_dimension=z_dim,
                              model_path=f"../models/RCVAE/{data_name}/{cell_type}/{z_dim}/", )

        network.restore_model()

        if sparse.issparse(data.X):
            data.X = data.X.A

        feed_data = data.X

        train_labels, _ = trvae.label_encoder(data)
        fake_labels = np.ones(train_labels.shape)

        latent_with_true_labels = network.to_z_latent(feed_data, train_labels)
        latent_with_fake_labels = network.to_z_latent(feed_data, fake_labels)
        mmd_latent_with_true_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=False)
        mmd_latent_with_fake_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=True)

        cell_type_ctrl = cell_type_adata.copy()[cell_type_adata.obs['condition'] == source_key]
        print(cell_type_ctrl.shape, cell_type_adata.shape)

        pred_celltypes = network.predict(cell_type_ctrl,
                                         encoder_labels=np.zeros((cell_type_ctrl.shape[0], 1)),
                                         decoder_labels=np.ones((cell_type_ctrl.shape[0], 1)))
        pred_adata = anndata.AnnData(X=pred_celltypes)
        pred_adata.obs['condition'] = ['predicted'] * pred_adata.shape[0]
        pred_adata.var = cell_type_adata.var

        if data_name == "pbmc":
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=100, method="wilcoxon")
            top_100_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_key].tolist()
            gene_list = top_100_genes[:10]
        else:
            sc.tl.rank_genes_groups(cell_type_adata, groupby="condition", n_genes=100, method="wilcoxon")
            top_50_down_genes = cell_type_adata.uns["rank_genes_groups"]["names"][source_key].tolist()
            top_50_up_genes = cell_type_adata.uns["rank_genes_groups"]["names"][target_key].tolist()
            top_100_genes = top_50_up_genes + top_50_down_genes
            gene_list = top_50_down_genes[:5] + top_50_up_genes[:5]

        cell_type_adata = cell_type_adata.concatenate(pred_adata)

        trvae.plotting.reg_mean_plot(cell_type_adata,
                                     top_100_genes=top_100_genes,
                                     gene_list=gene_list,
                                     condition_key='condition',
                                     axis_keys={"x": 'predicted', 'y': target_key},
                                     labels={'x': 'pred stim', 'y': 'real stim'},
                                     legend=False,
                                     fontsize=20,
                                     textsize=14,
                                     title=cell_type,
                                     path_to_save=os.path.join(path_to_save,
                                                               f'rcvae_reg_mean_{data_name}_{cell_type}.pdf'))

        trvae.plotting.reg_var_plot(cell_type_adata,
                                    top_100_genes=top_100_genes,
                                    gene_list=gene_list,
                                    condition_key='condition',
                                    axis_keys={"x": 'predicted', 'y': target_key},
                                    labels={'x': 'pred stim', 'y': 'real stim'},
                                    legend=False,
                                    fontsize=20,
                                    textsize=14,
                                    title=cell_type,
                                    path_to_save=os.path.join(path_to_save,
                                                              f'rcvae_reg_var_{data_name}_{cell_type}.pdf'))

        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
        latent_with_true_labels.obs['condition'] = data.obs['condition'].values
        latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
        latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
        latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
        mmd_latent_with_true_labels.obs['condition'] = data.obs['condition'].values
        mmd_latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
        mmd_latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
        mmd_latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

        color = ['condition', cell_type_key]

        sc.pp.neighbors(train_data)
        sc.tl.umap(train_data)
        sc.pl.umap(train_data, color=color,
                   save=f'_{data_name}_{cell_type}_train_data',
                   show=False)

        sc.pp.neighbors(latent_with_true_labels)
        sc.tl.umap(latent_with_true_labels)
        sc.pl.umap(latent_with_true_labels, color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_true_labels",
                   show=False)

        sc.pp.neighbors(latent_with_fake_labels)
        sc.tl.umap(latent_with_fake_labels)
        sc.pl.umap(latent_with_fake_labels, color=color,
                   save=f"_{data_name}_{cell_type}_latent_with_fake_labels",
                   show=False)

        sc.pp.neighbors(mmd_latent_with_true_labels)
        sc.tl.umap(mmd_latent_with_true_labels)
        sc.pl.umap(mmd_latent_with_true_labels, color=color,
                   save=f"_{data_name}_{cell_type}_mmd_latent_with_true_labels",
                   show=False)

        sc.pp.neighbors(mmd_latent_with_fake_labels)
        sc.tl.umap(mmd_latent_with_fake_labels)
        sc.pl.umap(mmd_latent_with_fake_labels, color=color,
                   save=f"_{data_name}_{cell_type}_mmd_latent_with_fake_labels",
                   show=False)

        sc.pl.violin(cell_type_adata, keys=top_100_genes[0], groupby='condition',
                     save=f"_{data_name}_{cell_type}_{top_100_genes[0]}",
                     show=False)

        plt.close("all")


def visualize_trained_network_results_multimodal(data_dict, z_dim=100):
    plt.close("all")
    data_name = data_dict.get('name', None)
    source_key = data_dict.get('source_key', None)
    target_key = data_dict.get('target_key', None)

    data = sc.read(f"../data/{data_name}/train_{data_name}.h5ad")
    path_to_save = f"../results/RCVAE/{data_name}/{z_dim}/{source_key} to {target_key}/Visualizations/"
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)

    network = trvae.trVAE(x_dimension=data.shape[1],
                          z_dimension=z_dim,
                          model_path=f"../models/RCVAE/{data_name}/{z_dim}/", )
    network.restore_model()
    if sparse.issparse(data.X):
        data.X = data.X.A

    feed_data = data.X
    train_labels, _ = trvae.label_encoder(data)
    fake_labels = np.ones(train_labels.shape)
    latent_with_true_labels = network.to_z_latent(feed_data, train_labels)
    latent_with_fake_labels = network.to_z_latent(feed_data, fake_labels)
    mmd_latent_with_true_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=False)
    mmd_latent_with_fake_labels = network.to_mmd_layer(network, feed_data, train_labels, feed_fake=True)

    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    latent_with_true_labels = sc.AnnData(X=latent_with_true_labels)
    latent_with_true_labels.obs['condition'] = data.obs['condition'].values
    # latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels)
    latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
    # latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    mmd_latent_with_true_labels = sc.AnnData(X=mmd_latent_with_true_labels)
    mmd_latent_with_true_labels.obs['condition'] = data.obs['condition'].values
    # mmd_latent_with_true_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    mmd_latent_with_fake_labels = sc.AnnData(X=mmd_latent_with_fake_labels)
    mmd_latent_with_fake_labels.obs['condition'] = data.obs['condition'].values
    # mmd_latent_with_fake_labels.obs[cell_type_key] = data.obs[cell_type_key].values

    color = ['condition']

    sc.pp.neighbors(data)
    sc.tl.umap(data)
    sc.pl.umap(data, color=color,
               save=f'_{data_name}_train_data',
               show=False)

    sc.pp.neighbors(latent_with_true_labels)
    sc.tl.umap(latent_with_true_labels)
    sc.pl.umap(latent_with_true_labels, color=color,
               save=f"_{data_name}_latent_with_true_labels",
               show=False)

    sc.pp.neighbors(latent_with_fake_labels)
    sc.tl.umap(latent_with_fake_labels)
    sc.pl.umap(latent_with_fake_labels, color=color,
               save=f"_{data_name}__latent_with_fake_labels",
               show=False)

    sc.pp.neighbors(mmd_latent_with_true_labels)
    sc.tl.umap(mmd_latent_with_true_labels)
    sc.pl.umap(mmd_latent_with_true_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_true_labels",
               show=False)

    sc.pp.neighbors(mmd_latent_with_fake_labels)
    sc.tl.umap(mmd_latent_with_fake_labels)
    sc.pl.umap(mmd_latent_with_fake_labels, color=color,
               save=f"_{data_name}_mmd_latent_with_fake_labels",
               show=False)
    plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    arguments_group = parser.add_argument_group("Parameters")
    arguments_group.add_argument('-d', '--data', type=str, required=True,
                                 help='name of dataset you want to train')
    arguments_group.add_argument('-z', '--z_dim', type=int, default=20, required=False,
                                 help='latent space dimension')
    arguments_group.add_argument('-m', '--mmd_dimension', type=int, default=128, required=False,
                                 help='MMD Layer dimension')
    arguments_group.add_argument('-a', '--alpha', type=float, default=0.005, required=False,
                                 help='Alpha coeff in loss term')
    arguments_group.add_argument('-b', '--beta', type=float, default=100, required=False,
                                 help='Beta coeff in loss term')
    arguments_group.add_argument('-k', '--kernel', type=str, default='multi-scale-rbf', required=False,
                                 help='Kernel type')
    arguments_group.add_argument('-n', '--n_epochs', type=int, default=5000, required=False,
                                 help='Maximum Number of epochs for training')
    arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                                 help='Batch Size')
    arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.2, required=False,
                                 help='Dropout ratio')
    arguments_group.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False,
                                 help='Learning rate of optimizer')
    arguments_group.add_argument('-y', '--early_stop_limit', type=int, default=50, required=False,
                                 help='do train the network')
    arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                                 help='Learning rate of optimizer')

    args = vars(parser.parse_args())

    data_dict = DATASETS[args['data']]
    if args['do_train'] == 1:
        del args['do_train']
        if args['data'] == 'multimodal':
            del args['data']
            train_network_multi(data_dict=data_dict, **args)
            visualize_trained_network_results_multimodal(data_dict=data_dict, z_dim=args['z_dim'])
        else:
            del args['data']
            train_network(data_dict=data_dict, **args)
            visualize_trained_network_results(data_dict=data_dict, z_dim=args['z_dim'])
    # reconstruct_whole_data(data_dict=data_dict, z_dim=args['z_dim'])
    # stacked_violin_plot(data_dict, method="RCVAE", score_type="median_score")
    # plot_boxplot(data_dict=data_dict, method="RCVAE", n_genes=50, restore=False,
    #              score_type="median_score", y_measure="AE:max(x, 1)", scale="normal")
    print(f"Model for {data_dict['name']} has been trained and sample results are ready!")
