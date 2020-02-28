import pandas as pd
import numpy as np
from scipy import sparse
import scipy as sp
import matplotlib.pylab as plt

# read the interaction matrix

#interaction_sparse_stage = sparse.load_npz('data/interaction_v3.npz')
# interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)


interaction_sparse_step = sparse.load_npz('data/interaction_v5.npz')
# interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

plt.spy(interaction_sparse_stage, precision='present',aspect='auto')
plt.spy(interaction_sparse_step,precision=0.1, aspect='auto')

plt.spy(interaction_sparse, aspect='auto', precision=0.1, markersize=1,marker=',').grid(b=None)
