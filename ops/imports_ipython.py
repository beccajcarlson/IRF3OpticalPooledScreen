# plotting for notebook and code reloading 

from ops.imports import *
from ops.firesnake import Snake
### BECCA
from ops.firedove import *
###

import IPython
IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
IPython.get_ipython().run_line_magic('autoreload', '2')


import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm_notebook as tqdn