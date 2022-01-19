import collections
from typing import Iterable, List

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scMVP.models.utils import one_hot


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        #use_batch_norm: bool = False,
        use_relu: bool = True,
        #use_relu: bool = False,
        bias: bool = True,
        RNA_mode = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.LayerNorm(n_out, eps=0.0001),
                            nn.LeakyReLU() if RNA_mode else None,
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.0001)
                            if use_batch_norm
                            else None,
                            nn.LeakyReLU() if use_relu else nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                    zip(layers_dim[:-1], layers_dim[1:])
                )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :param instance_id: Use a specific conditional instance normalization (batchnorm)
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x
# Classifer
class Classifer(nn.Module):
    def __init__(
                 self,
                 n_input: int,
                 n_output: int,
                 ):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(n_input,n_output),
            nn.Softmax(dim=-1)
        )
    def forward(self, z: torch.Tensor, *cat_list: int,):
        predict_z = self.classifer(z)
        return predict_z

# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent
# Encoder_nb
class Encoder_nb(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Encoder_nb_layers
class Encoder_nb_layers(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.prelayers = nn.Sequential(
            nn.Linear(n_input, 10*n_hidden), nn.Linear(10*n_hidden, 5*n_hidden),
            nn.Linear(5*n_hidden, n_hidden), nn.LeakyReLU()
        )
        self.encoder = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        pre_x = self.prelayers(x)
        q = self.encoder(pre_x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Encoder_peak_layers
class Encoder_peak_layers(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.prelayers = nn.Sequential(
            nn.Linear(n_input, 50*n_hidden), nn.Linear(50*n_hidden, 10*n_hidden),
            nn.Linear(10*n_hidden, n_hidden), nn.LeakyReLU()
        )
        self.encoder = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        pre_x = self.prelayers(x)
        q = self.encoder(pre_x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Encoder_nb_attention
class Encoder_nb_attention(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)*self.px_decoder_aux(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Encoder_nb_selfattention_layer
class Encoder_nb_selfattention_layer(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder1 = FCLayers(
            n_in=n_input,
            n_out=8*n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=8*n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.w_q1 = nn.Linear(8*n_hidden, 8*n_hidden)
        self.w_k1 = nn.Linear(8*n_hidden, 8*n_hidden)
        self.w_v1 = nn.Linear(8*n_hidden, 8*n_hidden)
        self.encoder3 = FCLayers(
            n_in=8 * n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.w_q3 = nn.Linear(n_hidden, n_hidden)
        self.w_k3 = nn.Linear(n_hidden, n_hidden)
        self.w_v3 = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder1(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q1(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        K = self.w_k1(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        V = self.w_v1(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q = torch.matmul(attention, V).view(q.shape[0], q.shape[1])
        q = self.encoder3(q, *cat_list)
        Q = self.w_q3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        K = self.w_k3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        V = self.w_v3(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q = torch.matmul(attention, V).view(q.shape[0], q.shape[1])

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Encoder_nb_selfattention
class Encoder_nb_selfattention(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.px_encoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(n_hidden, eps=0.0001)


        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int, ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        assert q.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        K = self.w_k(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        V = self.w_v(q).view(q.shape[0],self.n_heads, q.shape[1]//self.n_heads,-1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(q.shape[0], q.shape[1])

        q_m = self.mean_encoder(q_a)
        q_v = torch.exp(self.var_encoder(q_a)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# encoder_libary
class Encoder_l(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            RNA_mode=False,
            use_relu=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent
# encoder_mse
class Encoder_mse(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Linear(n_input,n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Multi-Encoder-nb
class Multi_Encoder_nb(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.concat1 = nn.Linear(2 * n_hidden, n_hidden)
        self.concat2 = nn.Linear(n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError("Input training data should be 2 data types(RNA and ATAC),"
                             "but input was only {}.format(x.__len__())"
                             )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!"
                             )

        q1 = self.scRNA_encoder(x[0], *cat_list)
        q2 = self.scATAC_encoder(x[1], *cat_list)
        q = self.concat2(self.concat1(torch.cat((q1, q2), 1)))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Multi-Encoder-nb-attention
class Multi_Encoder_nb_attention(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.RNA_encoder_aux = nn.Sequential(
            nn.Linear(RNA_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.ATAC_encoder_aux = nn.Sequential(
            nn.Linear(ATAC_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.concat = nn.Linear(2 * n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError("Input training data should be 2 data types(RNA and ATAC),"
                             "but input was only {}.format(x.__len__())"
                             )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!"
                             )

        q1 = self.scRNA_encoder(x[0], *cat_list)*self.RNA_encoder_aux(x[0])
        q2 = self.scATAC_encoder(x[1], *cat_list)*self.ATAC_encoder_aux(x[1])
        q = self.concat(torch.cat((q1, q2), 1))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Multi-Encoder-self-attention
class Multi_Encoder_nb_SelfAttention(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        n_heads: int = 8,

    ):
        super().__init__()
        self.n_heads = n_heads
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.RNA_encoder_aux = nn.Sequential(
            nn.Linear(RNA_input, n_hidden), nn.Linear(n_hidden, n_hidden), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)

        self.do = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(n_hidden, eps=0.0001)

        self.concat = nn.Linear(2 * n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError("Input training data should be 2 data types(RNA and ATAC),"
                             "but input was only {}.format(x.__len__())"
                             )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!"
                             )

        q1 = self.scRNA_encoder(x[0], *cat_list)*self.RNA_encoder_aux(x[0])
        q2 = self.scATAC_encoder(x[1], *cat_list)
        assert q2.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        K = self.w_k(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        V = self.w_v(q2).view(q2.shape[0], self.n_heads, q2.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q2 = torch.matmul(attention, V).view(q2.shape[0], q2.shape[1])

        q = self.concat(torch.cat((q1, q2), 1))
        q = self.layernorm(q)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

# Multi-Encoder
class Multi_Encoder(nn.Module):
    def __init__(
        self,
        RNA_input: int,
        ATAC_input,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.scRNA_encoder = FCLayers(
            n_in=RNA_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.scATAC_encoder = FCLayers(
            n_in=ATAC_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            RNA_mode=False,
        )
        self.concat1 = nn.Linear(2 * n_hidden, n_hidden)
        self.concat2 = nn.Linear(n_hidden, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: list, *cat_list: int):
        # Parameters for latent distribution
        if x.__len__() != 2:
            raise ValueError("Input training data should be 2 data types(RNA and ATAC),"
                             "but input was only {}.format(x.__len__())"
                             )
        if not torch.is_tensor(x[0]):
            raise ValueError("training data should be a tensor!"
                             )

        q1 = self.scRNA_encoder(x[0], *cat_list)
        q2 = self.scATAC_encoder(x[1], *cat_list)
        q = self.concat2(self.concat1(torch.cat((q1, q2), 1)))
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent
# Multi-Decoder-nb-log-peak
class Multi_Decoder_nb_log(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2* n_hidden), nn.Linear(2* n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, libary_scale = None, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = (self.cluster_decoder(gamma, *cat_list))
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)# libary_scale
        else:
            p_rna_rate = torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12) # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))


        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale,dim=-1)*self.px_atac_decoder_aux(z)# for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout

# Multi-Dncoder-nb-log RNA count
class Multi_Decoder_nb_log_peak(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2* n_hidden), nn.Linear(2* n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output), nn.Sigmoid()
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Softmax(dim=-1)
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, libary_scale = None, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = (self.cluster_decoder(gamma, *cat_list))
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)


        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)# libary_scale
        else:
            p_rna_rate = torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12) # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))


        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_scale = p_atac_scale*self.px_atac_decoder_aux(z)# for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_scale

        return p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout

# Multi-Dncoder-nb-selfattention
class Multi_Decoder_nb_SelfAttention(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            # release version 210228
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2* n_hidden), nn.Linear(2* n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )

        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )

        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )

        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output), nn.Sigmoid()
        )

        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Softmax(dim=-1)
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, libary_scale = None, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = self.cluster_decoder(gamma, *cat_list)
            #test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)

        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        if libary_scale is not None:
            p_rna_rate = torch.exp(libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)# libary_scale
        else:
            p_rna_rate = torch.exp(libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12) # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        assert p_atac.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        K = self.w_k(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        V = self.w_v(p_atac).view(p_atac.shape[0], self.n_heads, p_atac.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        p_atac = torch.matmul(attention, V).view(p_atac.shape[0], p_atac.shape[1])

        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))
        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_scale = p_atac_scale*self.px_atac_decoder_aux(z)# for zinp and zip loss

        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_scale
        else:
            p_atac_mean = torch.exp(libaray_atac) * p_atac_scale

        return p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout


# Multi-Dncoder-nb
class Multi_Decoder_nb(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean gamma
        if is_cluster:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2* n_hidden), nn.Linear(2* n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)
        # auxiliary decoder
        self.px_rna_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, RNA_output), nn.Sigmoid()
        )
        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, libary_scale = None, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        #print(gamma)
        if gamma is not None:
            cluster_temp = (self.cluster_decoder(gamma, *cat_list))
            #test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
            #release version 210228
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)


        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))



        if libary_scale is not None:
            p_rna_rate = (libary_scale) * p_rna_scale * self.px_rna_decoder_aux(z)# libary_scale
        else:
            p_rna_rate = (libaray_gene) * p_rna_scale * self.px_rna_decoder_aux(z)  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12) # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            # test version 210302
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))


        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale,dim=-1)*self.px_atac_decoder_aux(z)# for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout

# Multi-Dncoder
class Multi_Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        RNA_output: int,
        ATAC_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        is_cluster: bool = True,
        n_cluster: int = None
    ):
        super().__init__()

        # RNA-seq decoder
        self.scRNA_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        # mean gamma
        if is_cluster:
            # release version 210228
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, 2* n_hidden), nn.Linear(2* n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        else:
            self.rna_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, RNA_output), nn.Softmax(dim=-1)
            )
        # dispersion: here we only deal with gene-cell dispersion case
        self.rna_r_decoder = nn.Linear(n_hidden, RNA_output)
        # dropout
        self.rna_dropout_decoder = nn.Linear(n_hidden, RNA_output)

        # ATAC decoder
        self.scATAC_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        # mean possion
        if is_cluster:
            self.cluster_decoder = FCLayers(
                n_in=n_cluster,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
            )
        self.atac_scale_decoder = nn.Sequential(
            nn.Linear( n_hidden, n_hidden * 4), nn.Linear(n_hidden * 4, ATAC_output)
        )

        self.px_atac_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, ATAC_output), nn.Sigmoid()
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.atac_r_decoder = nn.Linear(n_hidden, ATAC_output)
        # dropout
        self.atac_dropout_decoder = nn.Linear(n_hidden, ATAC_output)

        # libaray scale for each cell
        self.libaray_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.libaray_rna_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden,1)
        )
        self.libaray_atac_scale_decoder =  nn.Sequential(
            nn.Linear(n_hidden,1)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor, *cat_list: int, libary_scale = None, gamma = None, libary_atac = None):
        # The decoder returns values for the parameters of the ZINB distribution of scRNA-seq
        p_rna = self.scRNA_decoder(z, *cat_list)
        libaray_temp = self.libaray_decoder(z_c, *cat_list)
        libaray_gene = self.libaray_rna_scale_decoder(libaray_temp)

        if gamma is not None:
            cluster_temp = (self.cluster_decoder(gamma, *cat_list))
            #test version 210302
            p_rna_scale = self.rna_scale_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))
        else:
            p_rna_scale = self.rna_scale_decoder(p_rna)


        p_rna_dropout = self.rna_dropout_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))



        if libary_scale is not None:
            p_rna_rate = torch.exp(libary_scale) * p_rna_scale # libary_scale
        else:
            p_rna_rate = torch.exp(libaray_gene) * p_rna_scale  # torch.clamp( , max=12)

        p_rna_rate.clamp(max=12) # maybe it is unnecessary
        p_rna_r = self.rna_r_decoder(torch.mul(p_rna, torch.sigmoid(cluster_temp)))

        # The decoder returns values for the parameters of the ZIP distribution of scATAC-seq
        p_atac = self.scATAC_decoder(z, *cat_list)
        if gamma is not None:
            p_atac_scale = self.atac_scale_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        else:
            p_atac_scale = self.atac_scale_decoder(torch.cat([p_atac, torch.softmax(libaray_temp, dim=-1)], dim=-1))

        p_atac_r = self.atac_r_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))

        p_atac_dropout = self.atac_dropout_decoder(torch.mul(p_atac, torch.sigmoid(cluster_temp)))


        libaray_atac = self.libaray_atac_scale_decoder(libaray_temp)
        p_atac_mean = torch.softmax(p_atac_scale,dim=-1) * self.px_atac_decoder_aux(z)# for zinp and zip loss
        if libary_atac is not None:
            p_atac_mean = torch.exp(libary_atac) * p_atac_mean

        return p_rna_scale, p_rna_r, p_rna_rate, p_rna_dropout, p_atac_scale, p_atac_r, p_atac_mean, p_atac_dropout

# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class DecoderSCVI_nb_rna(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden),nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = (library) * px_scale  # torch.clamp( , max=12) for scaled RNA data
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class DecoderSCVI_nb(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden),nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

# Decoder_nb_selfattention
class DecoderSCVI_nb_Selfattention(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden),nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])
        px_scale = self.px_scale_decoder(q_a)*self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class DecoderSCVI_Peak(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden),nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )
        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px) * self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

# Decoder_peak_selfattention
class DecoderSCVI_Peak_Selfattention(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden),nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(0.01)


        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Sigmoid()
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        
        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])
        px_scale = self.px_scale_decoder(q_a)*self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder_peak_selfattention_layer
class DecoderSCVI_Peak_Selfattention_Layer(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.px_decoder_aux = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        self.px_decoder1 = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.w_q1 = nn.Linear(n_hidden, n_hidden)
        self.w_k1 = nn.Linear(n_hidden, n_hidden)
        self.w_v1 = nn.Linear(n_hidden, n_hidden)

        self.px_decoder2 = FCLayers(
            n_in=n_hidden,
            n_out=8*n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=8*n_hidden,
            dropout_rate=0,
            RNA_mode=False,
        )
        self.w_q2 = nn.Linear(8*n_hidden, 8*n_hidden)
        self.w_k2 = nn.Linear(8*n_hidden, 8*n_hidden)
        self.w_v2 = nn.Linear(8*n_hidden, 8*n_hidden)

        self.do = nn.Dropout(0.01)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(8*n_hidden, n_output), nn.Sigmoid()
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(8*n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(8*n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder1(z, *cat_list)
        assert px.shape[1] % self.n_heads == 0, "n_heads cann't be divided by seq length!"
        Q = self.w_q1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v1(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        px = torch.matmul(attention, V).view(px.shape[0], px.shape[1])

        px = self.px_decoder2(px, *cat_list)
        Q = self.w_q2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        K = self.w_k2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        V = self.w_v2(px).view(px.shape[0], self.n_heads, px.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q_a = torch.matmul(attention, V).view(px.shape[0], px.shape[1])


        px_scale = self.px_scale_decoder(q_a)*self.px_decoder_aux(z)
        px_dropout = self.px_dropout_decoder(q_a)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(q_a) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class DecoderSCVI_mse(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = nn.Linear(n_input,n_hidden)

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, 2 * n_hidden), nn.Linear(2 * n_hidden, n_output)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super(LinearDecoderSCVI, self).__init__()

        # mean gamma
        self.n_batches = n_cat_list[0]  # Just try a simple case for now
        if self.n_batches > 1:
            self.batch_regressor = nn.Linear(self.n_batches - 1, n_output, bias=False)
        else:
            self.batch_regressor = None

        self.factor_regressor = nn.Linear(n_input, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_input, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        # The decoder returns values for the parameters of the ZINB distribution
        p1_ = self.factor_regressor(z)
        if self.n_batches > 1:
            one_hot_cat = one_hot(cat_list[0], self.n_batches)[:, :-1]
            p2_ = self.batch_regressor(one_hot_cat)
            raw_px_scale = p1_ + p2_
        else:
            raw_px_scale = p1_

        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v


class MultiEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent


class MultiDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list, instance_id=dataset_id)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout


class DecoderTOTALVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a linear decoder

    :param n_input: The dimensionality of the input (latent space)
    :param n_output_genes: The dimensionality of the output (gene space)
    :param n_output_proteins: The dimensionality of the output (protein space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    """

    def __init__(
        self,
        n_input: int,
        n_output_genes: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
    ):
        super().__init__()
        self.n_output_genes = n_output_genes
        self.n_output_proteins = n_output_proteins

        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden + n_input, n_output_genes), nn.Softmax(dim=-1)
        )

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # background mean parameters second decoder
        self.py_back_mean_log_alpha = nn.Linear(n_hidden + n_input, n_output_proteins)
        self.py_back_mean_log_beta = nn.Linear(n_hidden + n_input, n_output_proteins)

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden + n_input, n_output_proteins), nn.ReLU()
        )

        # dropout (mixture component for proteins, ZI probability for genes)
        self.sigmoid_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.px_dropout_decoder_gene = nn.Linear(n_hidden + n_input, n_output_genes)

        self.py_background_decoder = nn.Linear(n_hidden + n_input, n_output_proteins)

    def forward(self, z: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes
         #. Returns local parameters for the Mixture NB distribution for proteins

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary `py_` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. `scale` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         `back_alpha` and `back_beta` are the posterior parameters for `rate_back`.  `fore_scale` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

        :param z: tensor with shape ``(n_input,)``
        :param library_gene: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 3-tuple (first 2-tuple :py:class:`dict`, last :py:class:`torch.Tensor`)
        """
        px_ = {}
        py_ = {}

        px = self.px_decoder(z, *cat_list)
        px_cat_z = torch.cat([px, z], dim=-1)
        px_["scale"] = self.px_scale_decoder(px_cat_z)
        px_["rate"] = library_gene * px_["scale"]

        py_back = self.py_back_decoder(z, *cat_list)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z)
        py_["back_beta"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z))
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(z, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z) + 1
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(z, *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(p_mixing_cat_z)
        py_["mixing"] = self.py_background_decoder(p_mixing_cat_z)

        return (px_, py_, log_pro_back_mean)


# Encoder
class EncoderTOTALVI(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    :distribution: Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.z_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        def identity(x):
            return x

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary `latent` contains the samples of the latent variables, while `untran_latent`
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so `untran_latent["l"]` gives the normal sample that was later exponentiated to become `latent["l"]`.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        :param data: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 6-tuple. First 4 of :py:class:`torch.Tensor`, next 2 are `dict` of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(data, *cat_list)
        qz = self.z_encoder(q)
        qz_m = self.z_mean_encoder(qz)
        qz_v = torch.exp(self.z_var_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        ql_gene = self.l_gene_encoder(q)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = self.l_transformation(log_library_gene)

        latent = {}
        untran_latent = {}
        latent["z"] = z
        latent["l"] = library_gene
        untran_latent["z"] = untran_z
        untran_latent["l"] = log_library_gene

        return qz_m, qz_v, ql_m, ql_v, latent, untran_latent
