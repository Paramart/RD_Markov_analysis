import matplotlib.pyplot as plt
import numpy as np
import pyemma
import pickle
import warnings
warnings.filterwarnings('ignore')
from pyemma.util.contexts import settings
from deeptime.markov.tools.estimation import count_matrix
from deeptime.decomposition import cvsplit_trajs
from deeptime.decomposition._score import blocksplit_trajs
from pyemma._base.estimator import Estimator as estimator
from scipy.sparse import issparse
from pyemma.util.metrics import vamp_score

def score(hmm, dtrajs, score_method=None, score_k=None):
    K = hmm.transition_matrix
    C0t_train = hmm.count_matrix
    if issparse(K):
       K = K.toarray()
    if issparse(C0t_train):
       C0t_train = C0t_train.toarray()
    C00_train = np.diag(C0t_train.sum(axis=1))
    Ctt_train = np.diag(C0t_train.sum(axis=0))
    C0t_test_raw = count_matrix(dtrajs, hmm.lag, sparse_return=False)
    map_from = hmm.active_set[np.where(hmm.active_set < C0t_test_raw.shape[0])[0]]
    map_to = np.arange(len(map_from))
    C0t_test = np.zeros((hmm.nstates, hmm.nstates))
    C0t_test[np.ix_(map_to, map_to)] = C0t_test_raw[np.ix_(map_from, map_from)]
    C00_test = np.diag(C0t_test.sum(axis=1))
    Ctt_test = np.diag(C0t_test.sum(axis=0))
    return vamp_score(K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=score_k, score=score_method)

def scorecv(hmm, dtrajs, n=10, score_method=None, score_k=None):
    scores = []
    for i in range(n):
        dtrajs_split = blocksplit_trajs(dtrajs, blocksize=hmm.lag, sliding='sliding')
        dtrajs_train, dtrajs_test = cvsplit_trajs(dtrajs_split)
        estimator.fit(hmm, dtrajs_train)
        s = score(hmm, dtrajs_test, score_method=score_method, score_k=score_k)
        scores.append(s)
    return np.array(scores)

my_states = 3
n_clustercenters = [4, 7, 10, 25, 50, 75, 100]
tica_lag = 200
my_hmm_lag = 1

dim = 5
#PATH='write the path to your data file here'
PATH=''
with open(PATH + 'example_data.pickle', 'rb') as file:
         data = pickle.load(file)

tica = pyemma.coordinates.tica(data, lag=tica_lag, dim=dim, stride=1)
tica_output = tica.get_output()
tica_concatenated = np.concatenate(tica_output)

scores = np.zeros((len(n_clustercenters), my_states))
for n, z in enumerate(n_clustercenters):
    for m in range(my_states):
        with pyemma.util.contexts.settings(show_progress_bars=True):
             _cl = pyemma.coordinates.cluster_kmeans(tica_concatenated, k=z, max_iter=5000, stride=1)
             _hmm = pyemma.msm.estimate_hidden_markov_model(_cl.dtrajs, nstates=my_states, lag=my_hmm_lag, dt_traj='0.1 ns')
             try:
                 scores[n, m] = scorecv(_hmm, _cl.dtrajs, n=1, score_method='VAMP2', score_k=min(20, z))
             except:
                 continue

lower, upper = pyemma.util.statistics.confidence_interval(scores.T.tolist(), conf=0.95)
fig, ax = plt.subplots(figsize=(6, 4), sharey=False)
ax.fill_between(n_clustercenters, lower, upper, alpha=0.3)
ax.plot(n_clustercenters, np.mean(scores, axis=1), '-o')
ax.set_xlabel('Number of cluster centers')
ax.set_ylabel('VAMP2 score')
ax.set_title('Clustering', fontweight='normal')
fig.tight_layout()
plt.savefig('HMM_VAMP2_score_clustering.png', dpi=300)

