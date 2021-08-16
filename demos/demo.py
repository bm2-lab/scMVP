# testing the multi-modal vae method by scRNA-seq and scATAC data date: 04/24/2021
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, snareDataset, CellMeasurement, ATACDataset, geneDataset, GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
from scvi.inference import MultiPosterior, MultiTrainer
import torch
from scvi.models.multi_vae import Multi_VAE
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
import scanpy as sc
import anndata
import seaborn as sns
from scipy import stats


def allow_mmvae_for_test():
    print("Testing the basic tutorial mmvae")

test_mode = False
save_path = "data/"
n_epochs_all = None
show_plot = True

if not test_mode:
    save_path = "E:/data/qiliu/single-cell program/ATAC/snare data/"
# high variance gene list
hvg_data = pd.read_csv(save_path + 'snare_hvg.txt', header=None, index_col=None)
# high variance atac list
hvg_atac_data = pd.read_csv(save_path + 'snarep0_peak_var_mat_multipl.txt', header=None, index_col=None, sep="\t")
hvg_atac = hvg_atac_data.values[:,0]
hvg_atac_index = np.where(np.abs(hvg_atac_data.values[:,1])>0.009)[0]
hvg_atac = hvg_atac[hvg_atac_index]
# TF-IDF normalized atac peaks
dataset = pd.read_csv(save_path+'snare_atac_normalize_count.txt', header=0, index_col=0, sep="\t")
# atac dataloader
atac_dataset = GeneExpressionDataset()
cell_attributes_dict = {
    "barcodes": np.squeeze(np.asarray(dataset.columns.values, dtype=str))
    }
atac_dataset.populate_from_data(
    X=dataset.values.T, # notice the normalization
    batch_indices=None,
    gene_names=dataset.index.values,
    cell_attributes_dict=cell_attributes_dict,
    Ys=[],
)
#gene and atac joint dataloader
dataset = pd.read_csv(save_path+'snare_rna_normalize_count.txt', header=0, index_col=0, sep="\t")
gene_dataset = GeneExpressionDataset()
Ys = []
measurement = CellMeasurement(
        name="atac_expression",
        data=atac_dataset.X,
        columns_attr_name="atac_names",
        columns=atac_dataset.gene_names,
    )
Ys.append(measurement)
cell_attributes_dict = {
    "barcodes": np.squeeze(np.asarray(dataset.columns.values, dtype=str))
    }
gene_dataset.populate_from_data(
    X=dataset.values.T,
    batch_indices=None,
    gene_names=dataset.index.values,
    cell_attributes_dict=cell_attributes_dict,
    Ys=Ys,
)
high_count_genes = np.zeros(gene_dataset.X.shape[1])
for i in hvg_data.values[:,0]:
    high_count_genes[np.where(gene_dataset.gene_names == i)[0]] = 1
gene_dataset.update_genes(high_count_genes == 1)
gene_dataset.subsample_genes(new_n_genes=10000)
'''
# filter high variable atac data
high_count_atac = np.zeros(gene_dataset.atac_expression.shape[1])
for i in hvg_atac:
    high_count_atac[np.where(gene_dataset.atac_names== i)[0]] = 1
gene_dataset.atac_expression = gene_dataset.atac_expression[:, high_count_atac==1]
gene_dataset.atac_names = gene_dataset.atac_names[high_count_atac==1]
atac_dataset.update_genes(high_count_atac == 1)
'''
dataset = gene_dataset

# model para
n_epochs = 30 if n_epochs_all is None else n_epochs_all
lr = 5e-3
use_batches = False
use_cuda = True
n_centroids = 15
n_alfa = 1.0

# ATAC peak embedding
pre_atac_vae = VAE(atac_dataset.nb_genes, n_latent=20,n_batch=0, n_layers=1, log_variational=True, reconstruction_loss="nb")
pre_atac_trainer = UnsupervisedTrainer(
    pre_atac_vae,
    atac_dataset,
    train_size=0.9,
    use_cuda=use_cuda,
    frequency=5,
)
is_test_pragram = False
if is_test_pragram:
    pre_atac_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_atac_trainer.model.state_dict(), '%s/pre_atac_trainer_p0_210329_82.pkl' % save_path)
if os.path.isfile('%s/pre_atac_trainer_p0_210329_82.pkl' % save_path):
    pre_atac_trainer.model.load_state_dict(torch.load('%s/pre_atac_trainer_p0_210329_82.pkl' % save_path))
    pre_atac_trainer.model.eval()
else:
    pre_atac_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_atac_trainer.model.state_dict(), '%s/pre_atac_trainer_p0_210329_82.pkl' % save_path)
# ATAC pretrainer_posterior:
full = pre_atac_trainer.create_posterior(pre_atac_trainer.model, atac_dataset, indices=np.arange(len(atac_dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
prior_adata = anndata.AnnData(X=atac_dataset.X)
prior_adata.obsm["X_multi_vi"] = latent
prior_adata.obs['cell_type'] = torch.tensor(labels.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=30)
sc.tl.umap(prior_adata, min_dist=0.3)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
plt.show()
sc.tl.louvain(prior_adata)
sc.pl.umap(prior_adata, color=['louvain'])
plt.show()
# save data as csv file
df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=atac_dataset.barcodes )
df.insert(0,"labels",prior_adata.obs['louvain'].as_matrix())
df.to_csv(os.path.join(save_path,"scvi_atac_umap_210325_2.csv"))

df = pd.DataFrame(data=prior_adata.obsm["X_multi_vi"],  index=atac_dataset.barcodes)
df.to_csv(os.path.join(save_path,"scvi_latent_atac_imputation_210325_2.csv"))
imputed_values = full.sequential().imputation()
df = pd.DataFrame(data=imputed_values.T, columns=atac_dataset.barcodes, index=atac_dataset.gene_names)
#df.to_csv(os.path.join(save_path,"scvi_ATAC_imputation_210324_2.csv"))

# RNA embedding
pre_vae = VAE(dataset.nb_genes, n_latent=20,n_batch=0, n_layers=1, log_variational=True)
pre_trainer = UnsupervisedTrainer(
    pre_vae,
    dataset,
    train_size=0.9,
    use_cuda=use_cuda,
    frequency=5,
)
is_test_pragram = False
if is_test_pragram:
    pre_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer_p0_210329_82.pkl' % save_path)

if os.path.isfile('%s/pre_trainer_p0_210329_82.pkl' % save_path):
    pre_trainer.model.load_state_dict(torch.load('%s/pre_trainer_p0_210329_82.pkl' % save_path))
    pre_trainer.model.eval()
else:
    pre_trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(pre_trainer.model.state_dict(), '%s/pre_trainer_p0_210329_82.pkl' % save_path)
# RNA pretrainer_posterior:
full = pre_trainer.create_posterior(pre_trainer.model, dataset, indices=np.arange(len(dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
imputed_values = full.sequential().imputation()

df = pd.DataFrame(data=imputed_values.T, columns=dataset.barcodes, index=dataset.gene_names)
#df.to_csv(os.path.join(save_path,"gene_scvi_imputation_210324_2.csv"))
# visulization
prior_adata = anndata.AnnData(X=dataset.X)
prior_adata.obsm["X_multi_vi"] = latent
prior_adata.obs['cell_type'] = torch.tensor(labels.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=30)
sc.tl.umap(prior_adata, min_dist=0.3)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
plt.show()
sc.tl.louvain(prior_adata)
sc.pl.umap(prior_adata, color=['louvain'])
plt.show()

# save data as csv file
df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes )
df.insert(0,"labels",prior_adata.obs['louvain'].as_matrix())
df.to_csv(os.path.join(save_path,"scvi_umap_210325_2.csv"))

df = pd.DataFrame(data=prior_adata.obsm["X_multi_vi"],  index=dataset.barcodes)
df.to_csv(os.path.join(save_path,"scvi_latent_imputation_210325_2.csv"))

# joint RNA and ATAC embedding
multi_vae = Multi_VAE(dataset.nb_genes, len(dataset.atac_names), n_batch=0, n_latent=20, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
trainer = MultiTrainer(
    multi_vae,
    dataset,
    train_size=0.9,
    use_cuda=use_cuda,
    frequency=5,
)

clust_index_gmm = trainer.model.init_gmm_params(latent)
is_test_pragram = False
if is_test_pragram:
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), '%s/multi_vae_p0_210325_29_82_softmax.pkl' % save_path)

if os.path.isfile('%s/multi_vae_p0_210325_29_82_softmax.pkl' % save_path):
    trainer.model.load_state_dict(torch.load('%s/multi_vae_p0_210325_29_82_softmax.pkl' % save_path))
    trainer.model.eval()
else:
    trainer.model.RNA_encoder.load_state_dict(pre_trainer.model.z_encoder.state_dict())
    for param in trainer.model.RNA_encoder.parameters():
        param.requires_grad = False
    trainer.model.ATAC_encoder.load_state_dict(pre_atac_trainer.model.z_encoder.state_dict())
    for param in trainer.model.ATAC_encoder.parameters():
        param.requires_grad = False
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), '%s/multi_vae_p0_210325_29_82_softmax.pkl' % save_path)
# posterior
full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)
latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()
imputed_values = full.sequential().imputation()
# visulization
prior_adata = anndata.AnnData(X=latent)
prior_adata.obsm["X_multi_vi"] = latent
prior_adata.obs['cell_type'] = torch.tensor(labels.reshape(-1,1))
sc.pp.neighbors(prior_adata, use_rep="X_multi_vi", n_neighbors=30)
sc.tl.umap(prior_adata, min_dist=0.3)
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(prior_adata, color=["cell_type"], ax=ax, show=show_plot)
plt.show()
sc.tl.louvain(prior_adata)
sc.pl.umap(prior_adata, color=['louvain'])
plt.show()

# save file
df = pd.DataFrame(data=prior_adata.obsm["X_multi_vi"],  index=dataset.barcodes)
df.to_csv(os.path.join(save_path,"multivae_latent_imputation_rebedding_210329_2_softmax.csv"))

df = pd.DataFrame(data=prior_adata.obsm["X_umap"],  columns=["umap_dim1","umap_dim2"] , index=dataset.barcodes)
df.insert(0,"louvain",prior_adata.obs['louvain'].as_matrix())
df.to_csv(os.path.join(save_path,"multivae_umap_louvain_210329_2_softmax.csv"))

df = pd.DataFrame(data=imputed_values[1].T, columns=dataset.barcodes, index=dataset.atac_names)
#df.to_csv(os.path.join(save_path,"atac_multivae_imputation_210324_1_softmax.csv"))

df = pd.DataFrame(data=imputed_values[0].T, columns=dataset.barcodes, index=dataset.gene_names)
#df.to_csv(os.path.join(save_path,"gene_multivae_imputation_210324_1_softmax.csv"))
