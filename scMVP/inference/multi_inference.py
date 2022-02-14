from typing import Optional
import logging
import torch
from torch.distributions import Poisson, Gamma, Bernoulli, Normal
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import logsumexp
import torch.distributions as distributions
import numpy as np

from scMVP.inference import Posterior
from . import UnsupervisedTrainer

from scMVP.dataset import GeneExpressionDataset
from scMVP.models import multi_vae_attention
from sklearn.utils.linear_assignment_ import linear_assignment

logger = logging.getLogger(__name__)





class MultiPosterior(Posterior):
    r"""The functional data unit for Multivae. A `MultiPosterior` instance is instantiated with a model and
    a gene_dataset, and as well as additional arguments that for Pytorch's `DataLoader`. A subset of indices
    can be specified, for purposes such as splitting the data into train/test/validation. Each trainer instance of the `Trainer` class can therefore have multiple
    `MultiPosterior` instances to train a model. A `MultiPosterior` instance also comes with many methods or
    utilities for its corresponding data.


    :param model: A model instance from class ``Multivae``
    :param gene_dataset: A gene_dataset instance like ``ATACDataset()`` with attribute ``ATAC_expression``
    :param shuffle: Specifies if a `RandomSampler` or a `SequentialSampler` should be used
    :param indices: Specifies how the data should be split with regards to train/test or labelled/unlabelled
    :param use_cuda: Default: ``True``
    :param data_loader_kwarg: Keyword arguments to passed into the `DataLoader`

    Examples:

    Let us instantiate a `trainer`, with a gene_dataset and a model

        >>> gene_dataset = CbmcDataset()
        >>> totalvi = TOTALVI(gene_dataset.nb_genes, len(gene_dataset.protein_names),
        ... n_batch=gene_dataset.n_batches * False, n_labels=gene_dataset.n_labels, use_cuda=True)
        >>> trainer = TotalTrainer(vae, gene_dataset)
        >>> trainer.train(n_epochs=400)
    """

    def __init__(
        self,
        model: multi_vae_attention,
        gene_dataset: GeneExpressionDataset,
        shuffle: bool = False,
        indices: Optional[np.ndarray] = None,
        use_cuda: bool = True,
        data_loader_kwargs=dict(),
    ):

        super().__init__(
            model,
            gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=use_cuda,
            data_loader_kwargs=data_loader_kwargs,
        )
        # Add atac tensor as another tensor to be loaded
        self.data_loader_kwargs.update(
            {
                "collate_fn": gene_dataset.collate_fn_builder(
                    {"atac_expression": np.float32}# debug cell index
                )
            }
        )

        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    def corrupted(self):
        return self.update(
            {
                "collate_fn": self.gene_dataset.collate_fn_builder(
                    {"atac_expression": np.float32}, corrupted=True
                )
            }
        )

    def uncorrupted(self):
        return self.update(
            {
                "collate_fn": self.gene_dataset.collate_fn_builder(
                    {"atac_expression": np.float32}
                )
            }
        )

    @torch.no_grad()
    def elbo(self):
        elbo = self.compute_elbo(self.model)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo
    elbo.mode = "min"

    @torch.no_grad()
    def reconstruction_error(self):
        reconstruction_error = self.compute_reconstruction_error(self.model, self)
        logger.debug("Reconstruction Error : %.4f" % reconstruction_error)
        return reconstruction_error

    reconstruction_error.mode = "min"

    @torch.no_grad()
    def marginal_ll(self, n_mc_samples=1000):

        ll = self.compute_marginal_log_likelihood(self.model, self, n_mc_samples)
        logger.debug("True LL : %.4f" % ll)
        return ll

    def compute_elbo(self, vae:multi_vae_attention, **kwargs):
        """ Computes the ELBO.

        The ELBO is the reconstruction error + the KL divergences
        between the variational distributions and the priors.
        It differs from the marginal log likelihood.
        Specifically, it is a lower bound on the marginal log likelihood
        plus a term that is constant with respect to the variational distribution.
        It still gives good insights on the modeling of the data, and is fast to compute.
        """
        # Iterate once over the posterior and compute the elbo
        elbo = 0
        for i_batch, tensors in enumerate(self):
            (
                sample_batch_X,
                local_l_mean,
                local_l_var,
                batch_index,
                label,
                sample_batch_Y,
            ) = tensors

            reconst_loss, kl_divergence_local, kl_divergence_global = vae(
                sample_batch_X, sample_batch_Y, local_l_mean, local_l_var, batch_index, label
            )
            elbo += torch.sum(reconst_loss + kl_divergence_local).item()
        n_samples = len(self.indices)
        elbo += kl_divergence_global
        return elbo / n_samples

    def compute_reconstruction_error(self, vae:multi_vae_attention, **kwargs):
        r""" Computes log p(x/z), which is the reconstruction error .
                    Differs from the marginal log likelihood, but still gives good
                    insights on the modeling of the data, and is fast to compute

                    This is really a helper function to self.ll, self.ll_protein, etc.
                """
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors[
                                                                           :5
                                                                           ]  # general fish case

            # Distribution parameters
            outputs = vae.inference(sample_batch, batch_index, labels, **kwargs)
            p_rna_r = outputs["p_rna_r"]
            p_rna_rate = outputs["p_rna_rate"]
            p_rna_dropout = outputs["p_rna_dropout"]
            p_atac_mean = outputs["p_atac_mean"]
            p_atac_r = outputs["p_atac_r"]
            p_atac_dropout = outputs["p_atac_dropout"]

            # Reconstruction loss
            reconst_rna_loss = vae.get_reconstruction_loss(
                sample_batch,
                p_rna_rate,
                p_rna_r,
                p_rna_dropout,
#                bernoulli_params=bernoulli_params,
                **kwargs
            )
            reconst_atac_loss = vae.get_reconstruction_atac_loss(
                sample_batch,
                p_atac_mean,
                p_atac_r,
                p_atac_dropout,
                **kwargs
            )

            log_lkl += torch.sum(reconst_rna_loss).item()
            log_lkl += torch.sum(reconst_atac_loss).item()
        n_samples = len(self.indices)
        return log_lkl / n_samples

    def compute_marginal_log_likelihood(self, vae:multi_vae_attention , n_mc_samples):
        """ Computes a biased estimator for log p(x), which is the marginal log likelihood.

            Despite its bias, the estimator still converges to the real value
            of log p(x) when n_samples_mc (for Monte Carlo) goes to infinity
            (a fairly high value like 100 should be enough)
            Due to the Monte Carlo sampling, this method is not as computationally efficient
            as computing only the reconstruction loss
            """
        # Uses MC sampling to compute a tighter lower bound on log p(x)

        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
            to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

            for i in range(n_mc_samples):
                # Distribution parameters and sampled variables
                outputs = vae.inference(sample_batch, batch_index, labels)
                p_rna_r = outputs["p_rna_r"]
                p_rna_rate = outputs["p_rna_rate"]
                p_rna_dropout = outputs["p_rna_dropout"]
                qz_m = outputs["qz_m"]
                qz_v = outputs["qz_v"]
                z = outputs["z"]
                p_atac_mean = outputs["p_atac_mean"]
                p_atac_r = outputs["p_atac_r"]
                p_atac_dropout = outputs["p_atac_dropout"]
                mu_c = outputs["mu_c"]
                var_c = outputs["var_c"]
                gamma = outputs["gamma"]
                mu_c_max = outputs["mu_c_max"],
                var_c_max = outputs["var_c_max"],
                z_c_max = outputs["z_c_max"],

                # Reconstruction Loss
                reconst_rna_loss = vae.get_reconstruction_loss(
                    sample_batch,
                    p_rna_r,
                    p_rna_rate,
                    p_rna_dropout,
                )
                reconst_atac_loss = vae.get_reconstruction_atac_loss(
                    sample_batch,
                    p_atac_r,
                    p_atac_mean,
                    p_atac_dropout,
                )

                # Log-probabilities
                #p_l = Normal(local_l_mean, local_l_var.sqrt()).log_prob(library).sum(dim=-1)
                p_z = 0.0
                for prob, mu, var in mu_c, var_c, gamma:
                    p_z += prob*Normal(mu, var.sqrt()).log_prob(z).sum(dim=-1)

                p_x_zl = -reconst_rna_loss - reconst_atac_loss
                q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
                #q_z_max = Normal(mu_c_max, var_c_max.sqrt()).log_prob(z_c_max).sum(dim=-1)

                to_sum[:, i] = p_z + p_x_zl - q_z_x  #- q_z_max

            batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
            log_lkl += torch.sum(batch_log_lkl).item()

        n_samples = len(self.indices)
        # The minus sign is there because we actually look at the negative log likelihood
        return -log_lkl / n_samples

    @torch.no_grad()
    def get_latent(self, sample=False):
        """
        Output posterior z mean or sample, batch index, and label
        :param sample: z mean or z sample
        :return: three np.ndarrays, latent, batch_indices, labels
        """
        latent = []
        latent_rna = [];
        latent_atac = [];
        batch_indices = []
        labels = []
        cluster_gamma = []
        cluster_index = []
        for tensors in self:
            sample_batch_rna, local_l_mean, local_l_var, batch_index, label, sample_batch_atac = tensors
            give_mean = not sample
            latent_temp = self.model.sample_from_posterior_z(
                [sample_batch_rna, sample_batch_atac], y=label, give_mean=give_mean
            )
            latent += [
                latent_temp[0][0].cpu()
            ]
            latent_rna += [
                latent_temp[1][0].cpu()
            ]
            latent_atac += [
                latent_temp[2].cpu()
            ]
            gamma, mu_c, var_c, pi = self.model.get_gamma(latent_temp[0][0])
            cluster_gamma += [gamma.cpu()]
            cluster_index += [torch.argmax(gamma.cpu(),dim=1)]
            batch_indices += [batch_index.cpu()]
            labels += [label.cpu()]
        return (
            np.array(torch.cat(latent)),
            np.array(torch.cat(latent_rna)),
            np.array(torch.cat(latent_atac)),
            np.array(torch.cat(cluster_gamma)),
            np.array(torch.cat(cluster_index)),
            np.array(torch.cat(batch_indices)),
            np.array(torch.cat(labels)).ravel(),
        )

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 100,
        genes: Optional[np.ndarray] = None,
        batch_size: int = 256,
        #batch_size: int = 128,
    ) :
        """
        Create observation samples from the Posterior Predictive distribution

        :param n_samples: Number of required samples for each cell
        :param genes: Indices of genes of interest
        :param batch_size: Desired Batch size to generate data

        :return: Tuple (x_new, x_old)
            Where x_old has shape (n_cells, n_genes)
            Where x_new has shape (n_cells, n_genes, n_samples)
        """
        assert self.model.reconstruction_loss in ["zinb", "zip"]
        zero_inflated = "zinb"

        rna_old = []
        rna_new = []
        atac_old = []
        atac_new = []
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            outputs = self.model.inference(
                sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples
            )
            p_rna_r = outputs["p_rna_r"]
            p_rna_rate = outputs["p_rna_rate"]
            p_rna_dropout = outputs["p_rna_dropout"]
            p_atac_mean = outputs["p_atac_mean"]
            p_atac_dropout = outputs["p_atac_dropout"]

            # Generating rna-seq data
            p = p_rna_rate / (p_rna_rate + p_rna_r)
            r = p_rna_r
            # Important remark: Gamma is parametrized by the rate = 1/scale!
            l_train_rna = distributions.Gamma(concentration=r, rate=(1 - p) / p).sample()
            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train_rna = torch.clamp(l_train_rna, max=1e8)
            gene_expressions = distributions.Poisson(
                l_train_rna
            ).sample()  # Shape : (n_samples, n_cells_batch, n_genes)

            #Generating atac-seq data
            l_train_atac = torch.clamp(p_atac_mean, max=1e2)
            atac_expressions = distributions.Poisson(
                l_train_atac
            ).sample()

            # zero-inflate
            if zero_inflated:
                p_zero_rna = (1.0 + torch.exp(-p_rna_dropout)).pow(-1)
                random_prob_rna = torch.rand_like(p_zero_rna)
                gene_expressions[random_prob_rna <= p_zero_rna] = 0

                p_zero_atac = (1.0 + torch.exp(-p_atac_dropout)).pow(-1)
                random_prob_atac = torch.rand_like(p_zero_atac)
                atac_expressions[random_prob_atac <= p_zero_atac] = 0

            gene_expressions = gene_expressions.permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
            atac_expressions = atac_expressions.permute(
                [1, 2, 0]
            )

            rna_old.append(sample_batch[0].cpu())
            rna_new.append(gene_expressions.cpu())
            atac_old.append(sample_batch[1].cpu())
            atac_new.append(atac_expressions.cpu())

        rna_old = torch.cat(rna_old)  # Shape (n_cells, n_genes)
        rna_new = torch.cat(rna_new)  # Shape (n_cells, n_genes, n_samples)
        if genes is not None:
            gene_ids = self.gene_dataset.genes_to_index(genes)
            rna_new = rna_new[:, gene_ids, :]
            rna_old = rna_old[:, gene_ids]
        return rna_new.numpy(), rna_old.numpy(), atac_new.numpy(), rna_old.numpy()

    @torch.no_grad()
    def imputation(self, n_samples: int = 1):
        """ Gene imputation
        """
        imputed_rna_list = []
        imputed_atac_list = []
        label_list = []  # for the annotated data
        atac_list = []
        for tensors in self:
            x_rna, local_l_mean, local_l_var, batch_index, label, x_atac = tensors
            p_rna_rate, p_atac_rate = self.model.get_sample_rate(
                x=[x_rna,x_atac], batch_index=batch_index, y=label,  n_samples=n_samples, local_l_mean = local_l_mean, local_l_var = local_l_var
            )
            imputed_rna_list += [np.array(p_rna_rate.cpu())]
            imputed_atac_list += [np.array(p_atac_rate.cpu())]
            label_list += [np.array(label.cpu())] # only for annotated data
            atac_list += [np.array(x_atac.cpu())] # for the bins without call peak
        imputed_rna_list = np.concatenate(imputed_rna_list)
        imputed_atac_list = np.concatenate(imputed_atac_list)
        label_list = np.concatenate(label_list) # only for annotated data
        atac_list = np.concatenate(atac_list)# for the bins without call peak
        return imputed_rna_list.squeeze(), imputed_atac_list.squeeze(), label_list.squeeze(), atac_list

    @torch.no_grad()
    def get_sample_scale(self):
        p_rna_scales = []
        p_atac_scales = []
        for tensors in self:
            x_rna, _, _, batch_index, labels, x_atac = tensors
            p_rna_scales += [
                np.array(
                    (
                        self.model.get_sample_scale(
                            x=[x_rna,x_atac], batch_index=batch_index, y=labels, n_samples=1
                        )[0]
                    )
                )
            ]
            p_atac_scales += [
                np.array(
                    (
                        self.model.get_sample_scale(
                            x=[x_rna,x_atac], batch_index=batch_index, y=labels, n_samples=1
                        )[1]
                    )
                )
            ]
        return np.concatenate(p_rna_scales), np.concatenate(p_atac_scales)

    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, ind

    def get_clustering(self):
        latent, latent_rna, latent_atac, cluster_gamma, batch_indices, labels = self.get_latent()
        cluster_accuarcy, ind = self.cluster_acc(np.argmax(cluster_gamma,axis=1),labels)
        print('cell dataset multi-vae - clustering accuracy: %.2f%%' % (cluster_accuarcy * 100))
        return cluster_accuarcy, ind

class MultiTrainer(UnsupervisedTrainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``TOTALVI``
        :gene_dataset: A gene_dataset instance like ``CbmcDataset()`` with attribute ``protein_expression``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.93``.
        :test_size: The test size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.02``. Note that if train and test do not add to 1 the remainder is placed in a validation set
        :\*\*kwargs: Other keywords arguments from the general Trainer class.
    """
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        dataset,
        train_size=0.90,
        test_size=0.05,
        pro_recons_weight=1.0,
        n_epochs_back_kl_warmup=50, #200, init
        n_epochs_kl_warmup=200,
        **kwargs
    ):
        self.n_genes = dataset.nb_genes
        self.n_proteins = model.n_input_atac

        self.pro_recons_weight = pro_recons_weight
        self.n_epochs_back_kl_warmup = n_epochs_back_kl_warmup
        super().__init__(
            model, dataset, n_epochs_kl_warmup=n_epochs_kl_warmup, **kwargs
        )
        if type(self) is MultiTrainer:
            (
                self.train_set,
                self.test_set,
                self.validation_set,
            ) = self.train_test_validation(
                model, dataset, train_size, test_size, type_class=MultiPosterior
            )
            self.train_set.to_monitor = []
            self.test_set.to_monitor = ["elbo"]
            self.validation_set.to_monitor = ["elbo"]

    def loss(self, tensors):
        (
            sample_batch_X,
            local_l_mean,
            local_l_var,
            batch_index,
            label,
            sample_batch_Y,
        ) = tensors

        #reconst_loss, kl_divergence_local, kl_divergence_global = self.model(
        #    sample_batch_X, sample_batch_Y, local_l_mean, local_l_var, batch_index, label
        #)
        reconst_loss, kl_divergence_local, kl_divergence_global = self.model(
            sample_batch_X, sample_batch_Y, local_l_mean, local_l_var, batch_index, batch_index
        )
        loss = (
            self.n_samples
            * torch.mean(reconst_loss + self.back_warmup_weight * kl_divergence_local)
            + kl_divergence_global
        )
        print(
            "reconst_loss = %f,kl_divergence_local = %f,kl_weight = %f,loss = %f" %
              (torch.mean(reconst_loss), torch.mean(kl_divergence_local), self.back_warmup_weight, loss)
              )
        # self.KL_divergence = kl_divergence_global
        if self.normalize_loss:
            loss = loss / self.n_samples
        return loss


    def on_epoch_begin(self):
        super().on_epoch_begin()
        if self.n_epochs_back_kl_warmup is not None:
            #self.back_warmup_weight = min(1, self.epoch + self.n_epochs_back_kl_warmup / self.n_epochs_back_kl_warmup)
            self.back_warmup_weight = min(1, self.epoch + self.n_epochs_back_kl_warmup / self.n_epochs_back_kl_warmup)
        else:
            self.back_warmup_weight = 1.0

