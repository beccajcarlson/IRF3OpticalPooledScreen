from .AE import AE
from .AE46 import AE46
from .AE46_7ch import AE46_7ch
from .VAE import VAE
from .VAE46 import VAE46
from .augmentedAE import AugmentedAE

model_dict = {'AE': AE, 'AE46_7ch': AE46_7ch, 'AugmentedAE': AugmentedAE, 'VAE': VAE, 'AE46': AE46, 'VAE46': VAE46}
