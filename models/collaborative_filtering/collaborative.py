import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from lightfm.cross_validation import random_train_test_split
import scipy as sp
from scipy import sparse

# read the interaction matrix
#interaction_sparse = sparse.load_npz('data/interaction_v_binary.npz')
#interaction_sparse = sparse.load_npz('data/interaction_v4.npz') #latest recruiting step
#interaction_sparse = sparse.load_npz('data/interaction_v3.npz') #last recruiting stage
interaction_sparse = sparse.load_npz('data/interaction_v5.npz') # updated with client feedback
interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

# train test split for cv
train, test = random_train_test_split(interaction_sparse, test_percentage=0.3, random_state = None)

# create and train LightFM model
model = LightFM(loss='warp', item_alpha=1e-6, no_components=30)

model = model.fit(train, epochs=50, num_threads=4)

train_precision = precision_at_k(model, train, k=5).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(model, test, k=5).mean()
print('test precision at k: %s' %test_precision)

train_auc = auc_score(model, train, num_threads=4).mean()
print('train AUC: %s' %train_auc)
test_auc = auc_score(model, test, num_threads=4).mean()
print('test AUC: %s' %test_auc)
