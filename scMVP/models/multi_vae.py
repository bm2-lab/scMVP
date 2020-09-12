# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.distributions import Normal, kl_divergence as kl

from scMVP.models.log_likelihood import log_zinb_positive, log_nb_positive, log_zip_positive, binary_cross_entropy
from scMVP.models.modules import Encoder, DecoderSCVI, LinearDecoderSCVI, Multi_Encoder, Multi_Decoder, \
    reparameterize_gaussian
from scMVP.models.utils import one_hot

torch.backends.cudnn.benchmark = True


# VAE model
class Multi_VAE(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param mode: One of the following:
        * ``'vae'`` -single channel auto-encoder decoder neural framework for scRNA-seq data
        * ``'mm-vae'`` -multi-channels auto-encoder decoder neural framework for scRNA and scATAC data
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        RNA_input: int,
        ATAC_input: int = 0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_centroids: int = 20,
        n_alfa: float = 1.0,
        dropout_rate: float = 0.1,
        mode = "vae",
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
    ):
        super().__init__()
        self.mode = mode
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_input_atac = ATAC_input
        self.n_input_RNA = RNA_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_centroids = n_centroids
        self.alfa = n_alfa

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input, n_batch))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(RNA_input, n_labels))
            self.p_atac_r = torch.nn.Parameter(torch.randn(ATAC_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        if self.mode == "vae":
            # z encoder goes from the n_input-dimensional data to an n_latent-d
            # latent space representation
            self.z_encoder = Encoder(
                RNA_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            # l encoder goes from n_input-dimensional data to 1-d library size
            self.l_encoder = Encoder(
                RNA_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
            )
            # decoder goes from n_latent-dimensional space to n_input-d data
            self.decoder = DecoderSCVI(
                n_latent,
                RNA_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        elif self.mode == "mm-vae":
            if ATAC_input <= 0:
                raise ValueError("Input size of ATAC channel should be positive value,"
                                 "but input was {}.format(self.ATAC_input)"
                                 )

            # init c_params
            self.pi = nn.Parameter(torch.ones(n_centroids) / n_centroids)  # pc
            self.mu_c = nn.Parameter(torch.zeros(n_latent, n_centroids))  # mu
            self.var_c = nn.Parameter(torch.ones(n_latent, n_centroids))  # sigma^2

            self.RNA_encoder = Encoder(
                RNA_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.ATAC_encoder = Encoder(
                ATAC_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.RNA_ATAC_encoder = Multi_Encoder(
                RNA_input,
                ATAC_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.RNA_ATAC_decoder = Multi_Decoder(
                n_latent,
                RNA_input,
                ATAC_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            raise ValueError(
                "mode must be one of ['vae', 'mm-vae'"
                " ], but input was "
                "{}.format(self.mode)"
            )



    def get_latents(self, x_rna, y=None, x_atac=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z([x_rna, x_atac], y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=True):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x[0] = torch.log(1 + x[0])
            x[1] = torch.log(1 + x[1])
        #qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(x[0], y)
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(x[1], y)
        qz_m, qz_v, z = self.RNA_ATAC_encoder(x, y)
        if give_mean:
            z = qz_m,
            rna_z = qz_rna_m,
            atac_z = qz_atac_m
        return [z, rna_z, atac_z]

    def sample_from_posterior_l(self, x):
        r""" samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        outputs = self.inference(x=x, batch_index=batch_index, y=y, n_samples=n_samples)
        return outputs["p_rna_scale"], outputs["p_atac_scale"]


    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        outputs = self.inference(x=x, batch_index=batch_index, y=y, n_samples = n_samples)
        return outputs[ "p_rna_rate"], outputs["p_atac_mean"]

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout, **kwargs):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        return reconst_loss

    def get_reconstruction_atac_loss(self, x, mu, dispersion, dropout, type = "zinb", **kwargs):
        if type == "zinb":
            reconst_loss = -log_zinb_positive(x, mu, dispersion, dropout).sum(dim=-1)
        elif type == "zip":
            reconst_loss = - log_zip_positive(x, mu, dropout).sum(dim=-1)
        else:
            reconst_loss = - binary_cross_entropy(x, mu).sum(dim=-1)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch[0] = torch.log(1 + sample_batch[0])
            sample_batch[1] = torch.log(1 + sample_batch[1])

        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(sample_batch[0])
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(sample_batch[1])
        qz_m, qz_v, z = self.RNA_ATAC_encoder(sample_batch)

        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4.0 * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def init_gmm_params(self, z):
        """
        Init SCALE model with GMM model parameters
        """
        if z is None:
            raise("Input data is empty!")

        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        #z = self.encodeBatch(dataloader, device)
        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
        clust_index = gmm.predict(z)

        return clust_index


    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = torch.exp(self.pi.repeat(N, 1))  # NxK
        mu_c = self.mu_c.repeat(N, 1, 1)  # NxDxK
        var_c = torch.exp(self.var_c.repeat(N, 1, 1))  # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(
            torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c) ** 2 / (2 * var_c),
                                      dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi


    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if len(x_) != 2:
            raise ValueError("Input training data should be 2 data types(RNA and ATAC),"
                             "but input was only {}.format(len(x_))"
                             )
        x_rna = x_[0]
        x_atac = x_[1]
        if self.log_variational:
            x_rna = torch.log(1 + x_rna)
            x_atac = torch.log(1 + x_atac)


        # Sampling
        qz_rna_m, qz_rna_v, rna_z = self.RNA_encoder(x_rna, y)
        qz_atac_m, qz_atac_v, atac_z = self.ATAC_encoder(x_atac, y)
        qz_m, qz_v, z = self.RNA_ATAC_encoder([x_rna, x_atac], y)
        gamma, mu_c, var_c, pi = self.get_gamma(z)  # , self.n_centroids, c_params)
        index = torch.argmax(gamma,dim=1)
        #mu_c_max = torch.tensor([])
        #var_c_max = torch.tensor([])

        #for index1 in range(len(index)):
        #    mu_c_max = torch.cat((mu_c_max, mu_c[index1,:,index[index1]].float()),1)
        #    var_c_max = torch.cat((var_c_max, var_c[index1,:,index[index1]].float()),1)

        index1 = [i for i in range(len(index))]
        mu_c_max = mu_c[index1,:,index]
        var_c_max = var_c[index1,:,index]
        z_c_max = reparameterize_gaussian(mu_c_max, var_c_max)

        #decoder
        p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout\
        = self.RNA_ATAC_decoder(z, z_c_max, y)


        if self.dispersion == "gene-label":
            p_rna_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
            p_atac_r = F.linear(
                one_hot(y, self.n_labels), self.p_atac_r
            )
        elif self.dispersion == "gene-batch":
            p_rna_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            p_atac_r = F.linear(one_hot(batch_index, self.n_batch), self.p_atac_r)
        elif self.dispersion == "gene":
            p_rna_r = self.px_r
            p_atac_r = self.p_atac_r

        p_rna_r = torch.exp(p_rna_r)
        p_atac_r = torch.exp(p_atac_r)

        return dict(
            p_rna_scale=p_rna_scale,
            p_rna_r=p_rna_r,
            p_rna_rate=p_rna_rate,
            p_rna_dropout=p_rna_dropout,
            p_atac_scale=p_atac_scale,
            p_atac_r=p_atac_r,
            p_atac_mean=p_atac_mean,
            p_atac_dropout=p_atac_dropout,
            qz_rna_m=qz_rna_m,
            qz_rna_v=qz_rna_v,
            rna_z=rna_z,
            qz_atac_m=qz_atac_m,
            qz_atac_v=qz_atac_v,
            atac_z=atac_z,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            mu_c=mu_c,
            var_c=var_c,
            gamma=gamma,
            pi=pi,
            mu_c_max=mu_c_max,
            var_c_max=var_c_max,
            z_c_max=z_c_max,
        )

    def forward(self, x_rna, x_atac, local_l_mean, local_l_var, batch_index=None, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        x = [x_rna, x_atac]
        outputs = self.inference(x, batch_index, y)
        qz_rna_m = outputs["qz_rna_m"]
        qz_rna_v = outputs["qz_rna_v"]
        qz_atac_m = outputs["qz_atac_m"]
        qz_atac_v = outputs["qz_atac_v"]
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        p_rna_rate = outputs["p_rna_rate"]
        p_rna_r = outputs["p_rna_r"]
        p_rna_dropout = outputs["p_rna_dropout"]
        p_atac_r = outputs["p_atac_r"]
        p_atac_mean = outputs["p_atac_mean"]
        p_atac_dropout = outputs["p_atac_dropout"]
        mu_c = outputs["mu_c"]
        var_c = outputs["var_c"]
        gamma = outputs["gamma"]
        pi = outputs["pi"]

        """
           L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
                     = Eq(z,c|x)[ log p(x|z)] + Eq(z,c|x)[log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
           """
        n_centroids = pi.size(1)
        mu_expand = qz_m.unsqueeze(2).expand(qz_m.size(0), qz_m.size(1), n_centroids)
        logvar_expand = qz_v.unsqueeze(2).expand(qz_v.size(0), qz_v.size(1), n_centroids)

        # log p(z|c)
        logpzc = -0.5 * torch.sum(gamma * torch.sum(math.log(2 * math.pi) + \
                                                    torch.log(var_c) + \
                                                    torch.exp(logvar_expand) / var_c + \
                                                    (mu_expand - mu_c) ** 2 / var_c, dim=1), dim=1)
        # log p(c)
        logpc = torch.sum(gamma * torch.log(pi), 1)

        # log q(z|x) or q entropy
        qentropy = -0.5 * torch.sum(1 + qz_v + math.log(2 * math.pi), 1)

        # log q(c|x)
        logqcx = torch.sum(gamma * torch.log(gamma), 1)

        # kl(qz||pz)
        kld_qz_pz = -logpzc - logpc + qentropy + logqcx

        # kl(qz||qz_rna)
        kld_qz_rna = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(qz_rna_m, torch.sqrt(qz_rna_v))).sum(
            dim=1
        )

        # kl(qz||qz_atac)
        kld_qz_atac = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(qz_atac_m, torch.sqrt(qz_atac_v))).sum( #check the postive qz_v
            dim=1
        )

        # KL Divergence
        kl_divergence = kld_qz_pz + self.alfa * (kld_qz_rna + kld_qz_atac)

        #likelihood
        reconst_loss_rna = self.get_reconstruction_loss(x[0], p_rna_rate, p_rna_r, p_rna_dropout)
        reconst_loss_atac = self.get_reconstruction_atac_loss(x[1], p_atac_mean, p_atac_r, p_atac_dropout)  # implement this function
        reconst_loss = reconst_loss_rna + reconst_loss_atac
        # init the gmm model, training pc
        print("kld_qz_pz = %f,kld_qz_rna = %f,kld_qz_atac = %f,kl_divergence = %f,reconst_loss_rna = %f,\
        reconst_loss_atac = %f" % (torch.mean(kld_qz_pz), torch.mean(kld_qz_rna), torch.mean(kld_qz_atac),\
                                   torch.mean(kl_divergence), torch.mean(reconst_loss_rna), torch.mean(reconst_loss_atac)))
        return reconst_loss, kl_divergence, 0.0


class LDVAE(Multi_VAE):
    r"""Linear-decoded Variational auto-encoder model.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer (for encoder)
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log(data+1) prior to encoding for numerical stability. Not normalization.
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
    ):
        super().__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            log_variational,
            reconstruction_loss,
        )

        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def get_loadings(self):
        """ Extract per-gene weights (for each Z) in the linear decoder.
        """
        return self.decoder.factor_regressor.parameters()
