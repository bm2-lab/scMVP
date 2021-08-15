from .classifier import Classifier
# from .scanvi import SCANVI
from .vae import VAE, LDVAE
# from .autozivae import AutoZIVAE
# from .vaec import VAEC
# from .jvae import JVAE
# from .totalvi import TOTALVI
from .multi_vae import Multi_VAE
# from .multi_vae_v2 import Multi_VAE_v2
# from .multi_vae_v3 import Multi_VAE_v3
# from .multi_vae_v4 import Multi_VAE_v4
from .vaePeak import VAE_Peak
# from .multi_vae_v5 import Multi_VAE_v5
from .vae_attention import VAE_Attention
from .vaePeak_attention import VAE_Peak_Attention
# from .multi_vae_v6 import Multi_VAE_v6
from .vaePeak_selfattetion import VAE_Peak_SelfAttention
# from .vae_prelayer import VAE_PreLayer
# from .vaepeak_prelayer import VAE_Peak_PreLayer
from .multi_vae_attention import Multi_VAE_Attention


__all__ = [
    # "SCANVI",
    # "VAEC",
    "VAE",
    "VAE_Attention",
    "VAE_Peak",
    "VAE_Peak_Attention",
    "VAE_Peak_SelfAttention",
    # "VAE_PreLayer",
    # "VAE_Peak_PreLayer",
    "LDVAE",
    # "JVAE",
    "Classifier",
    # "AutoZIVAE",
    # "TOTALVI",
    "Multi_VAE",
    # "Multi_VAE_v2",
    # "Multi_VAE_v3",
    # "Multi_VAE_v4",
    # "Multi_VAE_v5",
    # "Multi_VAE_v6",
    "Multi_VAE_Attention",
]
