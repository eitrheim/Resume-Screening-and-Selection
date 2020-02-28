import pandas as pd
import numpy as np
import nltk
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from lightfm.cross_validation import random_train_test_split
import scipy as sp
import math
from scipy import sparse

# read the interaction matrix
interaction_sparse = sparse.load_npz('data/interaction_v_binary.npz')
interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

# train test split for cv
train, test = random_train_test_split(interaction_sparse, test_percentage=0.3, random_state = None)

### create and train LightFM model ###
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 50
ITEM_ALPHA = 1e-6
 
model = LightFM(loss='warp'
               , item_alpha=ITEM_ALPHA
               , no_components=NUM_COMPONENTS)


%time model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

%time train_precision = precision_at_k(model, train, k=5).mean()
print('train precision at k: %s' %train_precision)
%time test_precision = precision_at_k(model, test, k=5).mean()
print('test precision at k: %s' %test_precision)

%time train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
%time test_auc = auc_score(model, test, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)
