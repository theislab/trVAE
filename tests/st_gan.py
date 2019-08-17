import argparse
import os
from random import shuffle

import numpy as np
import scanpy.api as sc
import tensorflow as tf
from scipy import sparse

import trvae

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


### helper function


def predict(ctrl):
    pred = sess.run(gen_stim_fake, feed_dict={X_ctrl: ctrl, is_training: False})
    return pred


def low_embed(all):
    pred = sess.run(disc_c, feed_dict={X_ctrl: all, is_training: False})
    return pred


def low_embed_stim(all):
    pred = sess.run(disc_s, feed_dict={X_stim: all, is_training: False})
    return pred


# network

def discriminator_stimulated(tensor, reuse=False, ):
    with tf.variable_scope("discriminator_s", reuse=reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)
        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)

        return h, disc


def discriminator_control(tensor, reuse=False, ):
    with tf.variable_scope("discriminator_b", reuse=reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)
        return h, disc


def generator_stim_ctrl(image, reuse=False):
    with tf.variable_scope("generator_sb", reuse=reuse):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)
        return h


def generator_ctrl_stim(image, reuse=False, ):
    with tf.variable_scope("generator_bs", reuse=reuse):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)

        return h


def train(n_epochs, initial_run=True, global_step=None):
    if initial_run:
        print("Initial run")
        print("Training started")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        batch_ind1 = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
        mb_ctrl = train_real_ctrl[batch_ind1, :]
        mb_stim = train_real_stim[batch_ind1, :]
        for gen_it in range(2):
            _, g_loss, d_loss = sess.run([update_G, gen_loss, disc_loss],
                                         feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
        _, g_loss, d_loss = sess.run([update_G, gen_loss, disc_loss],
                                     feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
        print(f"Iteration {it}: {g_loss + d_loss}")
        _ = sess.run(update_D, feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print(f"Training finished")


def restore():
    saver.restore(sess, model_to_use)


# generator and discriminator

# =============================== downloading training and validation files ====================================
parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')

arguments_group = parser.add_argument_group("Parameters")
arguments_group.add_argument('-d', '--data', type=str, required=True,
                             help='name of dataset you want to train')
arguments_group.add_argument('-z', '--z_dim', type=int, default=100, required=False,
                             help='latent space dimension')
arguments_group.add_argument('-n', '--n_epochs', type=int, default=1000, required=False,
                             help='Maximum Number of epochs for training')
arguments_group.add_argument('-c', '--batch_size', type=int, default=512, required=False,
                             help='Batch Size')
arguments_group.add_argument('-t', '--do_train', type=int, default=1, required=False,
                             help='Batch Size')
arguments_group.add_argument('-r', '--dropout_rate', type=float, default=0.5, required=False,
                             help='Dropout ratio')

args = vars(parser.parse_args())
data_dict = DATASETS[args['data']]
data_name = data_dict.get('name', None)
cell_type_key = data_dict.get("cell_type", None)
source_key = data_dict.get('source_key')
target_key = data_dict.get('target_key')

train_path = f"../data/{data_name}/train_{data_name}.h5ad"
valid_path = f"../data/{data_name}/valid_{data_name}.h5ad"

data = sc.read(train_path)
validation = sc.read(valid_path)

if sparse.issparse(data.X):
    data.X = data.X.A
if sparse.issparse(validation.X):
    validation.X = validation.X.A

# =============================== data gathering ====================================
spec_cell_types = data_dict.get('spec_cell_types', None)
cell_types = data.obs[cell_type_key].unique().tolist()

for spec_cell_type in spec_cell_types:
    train_real = data.copy()[~((data.obs['condition'] == target_key) & (data.obs[cell_type_key] == spec_cell_type))]
    train_real_stim = train_real[train_real.obs["condition"] == target_key]
    train_real_ctrl = train_real[train_real.obs["condition"] == source_key]
    train_real_stim = train_real_stim.X

    ind_list = [i for i in range(train_real_stim.shape[0])]
    shuffle(ind_list)
    train_real_stim = train_real_stim[ind_list, :]

    gex_size = train_real_stim.shape[1]
    train_real_ctrl = train_real_ctrl.X
    ind_list = [i for i in range(train_real_ctrl.shape[0])]
    shuffle(ind_list)
    train_real_ctrl = train_real_ctrl[ind_list, :]

    eq = min(len(train_real_ctrl), len(train_real_stim))
    stim_ind = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
    ctrl_ind = np.random.choice(range(len(train_real_ctrl)), size=eq, replace=False)
    ##  selecting equal size for both stimulated and control cells
    train_real_ctrl = train_real_ctrl[ctrl_ind, :]
    train_real_stim = train_real_stim[stim_ind, :]

    # =============================== parameters ====================================
    model_to_use = f"../models/STGAN/{data_name}/{spec_cell_type}/stgan"
    os.makedirs(model_to_use, exist_ok=True)
    X_dim = gex_size
    z_dim = args['z_dim']
    h_dim = 200
    batch_size = args['batch_size']
    inflate_to_size = 100
    lambda_l2 = .8
    arch = {"noise_input_size": z_dim, "inflate_to_size": inflate_to_size,
            "epochs": 0, "bsize": batch_size, "disc_internal_size ": h_dim, "#disc_train": 1}
    X_stim = tf.placeholder(tf.float32, shape=[None, X_dim], name="data_stim")
    X_ctrl = tf.placeholder(tf.float32, shape=[None, X_dim], name="data_ctrl")
    time_step = tf.placeholder(tf.int32)
    size = tf.placeholder(tf.int32)
    learning_rate = 0.001
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    is_training = tf.placeholder(tf.bool)
    dr_rate = args['dropout_rate']
    const = 5
    gen_stim_fake = generator_ctrl_stim(X_ctrl)
    gen_ctrl_fake = generator_stim_ctrl(X_stim)

    recon_ctrl = generator_stim_ctrl(gen_stim_fake, reuse=True)
    recon_stim = generator_ctrl_stim(gen_ctrl_fake, reuse=True)

    disc_ctrl_fake, _ = discriminator_control(gen_ctrl_fake)
    disc_stim_fake, _ = discriminator_stimulated(gen_stim_fake)

    disc_ctrl_real, disc_c = discriminator_control(X_ctrl, reuse=True)
    disc_stim_real, disc_s = discriminator_stimulated(X_stim, reuse=True)

    # computing loss

    const_loss_s = tf.reduce_sum(tf.losses.mean_squared_error(recon_ctrl, X_ctrl))
    const_loss_b = tf.reduce_sum(tf.losses.mean_squared_error(recon_stim, X_stim))

    gen_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_fake - 1)) / 2
    gen_stim_loss = tf.reduce_sum(tf.square(disc_stim_fake - 1)) / 2

    disc_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_real - 1) + tf.square(disc_ctrl_fake)) / 2
    disc_stim_loss = tf.reduce_sum(tf.square(disc_stim_real - 1) + tf.square(disc_stim_fake)) / 2

    gen_loss = const * (const_loss_s + const_loss_b) + gen_ctrl_loss + gen_stim_loss
    disc_loss = disc_ctrl_loss + disc_stim_loss

    # applying gradients

    gen_sb_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_sb")
    gen_bs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_bs")
    dis_s_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_s")
    dis_b_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_b")
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        update_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss,
                                                                                var_list=dis_s_variables + dis_b_variables,
                                                                                )
        update_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss,
                                                                                var_list=gen_sb_variables + gen_bs_variables)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=1)
    init = tf.global_variables_initializer().run()

    path_to_save = f"../results/STGAN/{data_name}/{spec_cell_type}"
    sc.settings.figdir = path_to_save
    sc.settings.writedir = path_to_save
    do_train = args['do_train']
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    if do_train == 1:
        train(args['n_epochs'], initial_run=True, global_step=global_step)
    else:
        restore()

    print("model has been trained/restored!")

    CD4T = data.copy()[data.obs[cell_type_key] == spec_cell_type]
    ctrl_CD4T = data.copy()[(data.obs[cell_type_key] == spec_cell_type) & (data.obs['condition'] == source_key)]
    stim_CD4T = data.copy()[(data.obs[cell_type_key] == spec_cell_type) & (data.obs['condition'] == target_key)]

    if data_name == "pbmc":
        sc.tl.rank_genes_groups(CD4T, groupby="condition", n_genes=100, method="wilcoxon")
        top_100_genes = CD4T.uns["rank_genes_groups"]["names"][target_key].tolist()
        gene_list = top_100_genes[:10]
    else:
        sc.tl.rank_genes_groups(CD4T, groupby="condition", n_genes=100, method="wilcoxon")
        top_50_down_genes = CD4T.uns["rank_genes_groups"]["names"][source_key].tolist()
        top_50_up_genes = CD4T.uns["rank_genes_groups"]["names"][target_key].tolist()
        top_100_genes = top_50_up_genes + top_50_down_genes
        gene_list = top_50_down_genes[:5] + top_50_up_genes[:5]

    if sparse.issparse(ctrl_CD4T.X):
        ctrl_CD4T.X = ctrl_CD4T.X.A
        stim_CD4T.X = stim_CD4T.X.A

    predicted_cells = predict(ctrl_CD4T.X)
    all_Data = sc.AnnData(np.concatenate([ctrl_CD4T.X, stim_CD4T.X, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * len(ctrl_CD4T.X) + ["real_stim"] * len(stim_CD4T.X) + \
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = CD4T.var_names
    # all_Data.write("../data/reconstructed/CGAN/cgan_cd4t.h5ad")
    low_dim = low_embed_stim(train_real.X)
    dt = sc.AnnData(low_dim)
    sc.pp.neighbors(dt)
    sc.tl.umap(dt)
    dt.obs["cell_type"] = train_real.obs[cell_type_key]
    dt.obs["condition"] = train_real.obs["condition"]
    sc.pl.umap(dt, color=["cell_type"], show=False, frameon=False
               , save="_latent_cell_type.pdf")

    sc.pl.umap(dt, color=["condition"], show=False, frameon=False
               , save="_latent_condition.pdf", palette=["#96a1a3", "#A4E804"])

    trvae.plotting.reg_mean_plot(all_Data,
                                 top_100_genes=top_100_genes,
                                 gene_list=gene_list,
                                 condition_key='condition',
                                 axis_keys={"x": 'pred_stim', 'y': "real_stim"},
                                 labels={'x': 'pred stim', 'y': 'real stim'},
                                 legend=False,
                                 fontsize=20,
                                 textsize=14,
                                 title=spec_cell_type,
                                 path_to_save=os.path.join(path_to_save,
                                                          f'mmd_resnet_reg_mean_{data_name}_{spec_cell_type}.pdf'))

    trvae.plotting.reg_var_plot(all_Data,
                                top_100_genes=top_100_genes,
                                gene_list=gene_list,
                                condition_key='condition',
                                axis_keys={"x": 'pred_stim', 'y': "real_stim"},
                                labels={'x': 'pred stim', 'y': 'real stim'},
                                legend=False,
                                fontsize=20,
                                textsize=14,
                                title=spec_cell_type,
                                path_to_save=os.path.join(path_to_save,
                                                          f'mmd_resnet_reg_var_{data_name}_{spec_cell_type}.pdf'))

    # os.rename(src=os.path.join(path_to_save, "umap_latent_cell_type.png"),
    #           dst=os.path.join(path_to_save, f"SupplFig4b_style_transfer_celltype.png"))
    #
    # os.rename(src=os.path.join(path_to_save, "umap_latent_condition.png"),
    #           dst=os.path.join(path_to_save, f"SupplFig4b_style_transfer_condition.png"))

    tf.reset_default_graph()
