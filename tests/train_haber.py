import numpy as np
import scanpy as sc

import trvae


def predict_transition(adata, source_cond, target_cond, cell_type):
    source_adata = adata[adata.obs[condition_key] == source_cond]

    source_cond_key = source_cond.split("_to_")[-1]
    source_labels = np.zeros(source_adata.shape[0]) + label_encoder[source_cond_key]
    target_labels = np.zeros(source_adata.shape[0]) + label_encoder[target_cond]

    pred_target = network.predict(source_adata,
                                  encoder_labels=source_labels,
                                  decoder_labels=target_labels,
                                  )

    pred_adata = sc.AnnData(X=pred_target)
    pred_adata.obs[condition_key] = [source_cond + "_to_" + target_cond] * pred_target.shape[0]
    pred_adata.obs[cell_type_key] = [cell_type] * pred_target.shape[0]
    pred_adata.var_names = source_adata.var_names

    adata = adata.concatenate(pred_adata)
    return adata


adata = sc.read("./data/haber/haber.h5ad")
cell_type_key = 'cell_label'
condition_key = 'condition'

target_conditions = ['Hpoly.Day3', 'Hpoly.Day10', 'Salmonella']

train_adata, valid_adata = trvae.utils.train_test_split(adata, 0.80)
cell_types = adata.obs[cell_type_key].unique().tolist()

for cell_type in cell_types:
    net_train_adata = train_adata[~((train_adata.obs[cell_type_key] == cell_type) &
                                    (train_adata.obs[condition_key].isin(target_conditions)))]
    net_valid_adata = valid_adata[~((valid_adata.obs[cell_type_key] == cell_type) &
                                    (valid_adata.obs[condition_key].isin(target_conditions)))]

    network = trvae.archs.trVAEMulti(x_dimension=net_train_adata.shape[1],
                                     z_dimension=60,
                                     mmd_dimension=128,
                                     n_conditions=len(net_train_adata.obs[condition_key].unique()),
                                     alpha=1e-6,
                                     beta=10,
                                     eta=10,
                                     clip_value=1000,
                                     lambda_l1=0.0,
                                     lambda_l2=0.0,
                                     learning_rate=0.001,
                                     model_path=f"./models/trVAEMulti/best/haber-{cell_type}/",
                                     dropout_rate=0.2,
                                     output_activation='relu')

    label_encoder = {'Control': 0, 'Hpoly.Day3': 1, 'Hpoly.Day10': 2, 'Salmonella': 3}

    network.train(net_train_adata,
                  net_valid_adata,
                  label_encoder,
                  condition_key,
                  n_epochs=10000,
                  batch_size=512,
                  verbose=2,
                  early_stop_limit=750,
                  lr_reducer=0,
                  shuffle=True,
                  save=True,
                  )

    cell_type_adata = train_adata[train_adata.obs[cell_type_key] == cell_type]

    recon_adata = predict_transition(cell_type_adata, "Control", "Hpoly.Day3", cell_type)
    recon_adata = recon_adata.concatenate(predict_transition(recon_adata, "Control", "Hpoly.Day10", cell_type))
    recon_adata = recon_adata.concatenate(predict_transition(recon_adata, "Control", "Salmonella", cell_type))
    recon_adata = recon_adata.concatenate(predict_transition(recon_adata, "Hpoly.Day3", "Hpoly.Day10", cell_type))
    recon_adata = recon_adata.concatenate(
        predict_transition(recon_adata, "Control_to_Hpoly.Day3", "Hpoly.Day10", cell_type))

    recon_adata.write_h5ad(f"./data/reconstructed/trVAEMulti/Haber/{cell_type}.h5ad")
