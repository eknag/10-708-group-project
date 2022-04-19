"""
In this model are gathered some predefined neural nets architectures that may be used on 
benchmark datasets such as MNIST, CIFAR or CELEBA
"""

from .celeba import (
    Encoder_VAE_CELEBA,
    Decoder_AE_CELEBA,
)

from .cifar import (
    Encoder_VAE_CIFAR,
    Decoder_AE_CIFAR,
)

from .mnist import (
    Encoder_VAE_MNIST,
    Decoder_AE_MNIST,
)

__all__ = [
    "Encoder_VAE_CELEBA",
    "Decoder_AE_CELEBA",
    "Encoder_VAE_CIFAR",
    "Decoder_AE_CIFAR",
    "Encoder_VAE_MNIST",
    "Decoder_AE_MNIST",
]
