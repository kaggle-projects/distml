import itertools
import os
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from more_itertools import chunked_even
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump, load, hash as jhash
from sklearn.base import TransformerMixin, BaseEstimator


def get_random_dist(seed=None):
    rng = np.random.default_rng(seed)
    nsamples = int(rng.choice([
        np.exp(rng.random() * 2 + 3),
        np.exp(rng.random() * 1.5 + 3.5),
        np.exp(rng.random() * 1.2 + 2),
    ]))
    dist_model = [
        lambda t: rng.normal(t[0], abs(t[1])) + t[2],
        lambda t: rng.random() * t[0] + t[1],
        lambda t: rng.poisson(abs(t[0])) + t[1],
        lambda t: rng.beta(abs(t[0]), abs(t[1])) * t[2] + t[3],
    ]
    nr_dists = int(rng.random() * 5 + 5)
    dists = [(rng.choice(dist_model), (rng.random() * 4 - 2, rng.random() * 4 - 2, rng.random() * 4 - 2, rng.random() * 4 - 2)) for d in range(nr_dists)]
    weights = np.power(np.arange(len(dists)) + rng.beta(1, 5) * len(dists), 5)
    weights /= sum(weights)
    rng.shuffle(weights)
    a = []
    for i in range(nsamples):
        d, p = rng.choice(dists, p=weights)
        a.append(d(p))
    return a


def get_score(yes, no):
    def _get_fisher(cut):
        a, b, c, d = 0, 0, 0, 0
        for x in yes:
            if x > cut:
                a += 1
            else:
                c += 1
        for x in no:
            if x > cut:
                b += 1
            else:
                d += 1
        pvalue = stats.fisher_exact([[a, b], [c, d]], alternative='greater')[1]
        return pvalue

    def _get_cuts():
        occurence = defaultdict(str)
        for x in set(yes):
            occurence[x] += 'p'
        for x in set(no):
            occurence[x] += 'n'
        occurence = sorted(occurence.items())
        for i in range(len(occurence) - 1):
            a1, b1 = occurence[i]
            a2, b2 = occurence[i + 1]
            if b1 == b2 and len(b1) == 1:
                continue
            yield (a1 + a2) / 2

    pfisher = min(_get_fisher(cut) for cut in _get_cuts())
    pttest = stats.ttest_1samp(yes, np.median([*yes, *no]), alternative='greater').pvalue
    s = stats.gmean([pfisher, pttest])
    if np.isnan(s):
        s = 1.0
    s = np.clip(s, 0, 1)
    return np.clip(np.interp(-np.log10(s), [0, 1, 5, 100], [0, 60, 90, 100]) + np.random.normal(0, 2), 0, 100)

def dump_data():
    def do_job():
        dists = []
        for i in range(1000):
            d = [get_random_dist(), get_random_dist()]
            dists.append([*d, get_score(d[0], d[1])])
        os.makedirs('data', exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='', suffix='.job', dir='data', delete=False) as f:
            dump(dists, f.name)
    Parallel(n_jobs=-1)(delayed(do_job)() for i in range(320))


def load_data(n=-1):
    def load_1(filenames):
        ret = []
        for f in filenames:
            ds = load(f'data/{f}')
            rows = []
            for i, d in enumerate(ds):
                rows.append([jhash((f, i)), *d])
            ret.append(rows)
        return ret
    data_files = os.listdir('data')
    if n >= 0:
        data_files = data_files[:n]
    data = Parallel(n_jobs=-1)(delayed(load_1)(fs) for fs in chunked_even(data_files, 64))
    data = list(itertools.chain.from_iterable(itertools.chain.from_iterable(data)))
    return pd.DataFrame(data, columns=['uid', 'yes_data', 'no_data', 'score'])


def split_data(data_df, train_ratio, test_ratio):
    blen = 6
    max_num = int('f'*blen, 16)
    r = train_ratio/(train_ratio+test_ratio)
    is_train = data_df['uid'].apply(lambda x: int(x[:blen], 16) < r*max_num)
    return data_df[is_train], data_df[~is_train]


class GetStats(TransformerMixin, BaseEstimator):
    def __init__(
            self,
            use_std=True,
            use_var=True,
            use_mean=True,
            use_cnt=True,
            use_moments=4,
            use_skew_bias=True,
            use_skew_nobias=True,
            use_kurtosis_fisher_bias=True,
            use_kurtosis_fisher_nobias=True,
            use_kurtosis_pearson_bias=True,
            use_kurtosis_pearson_nobias=True,
            use_quantile=9,
            n_jobs=1,
    ):
        self.use_std = use_std
        self.use_var = use_var
        self.use_mean = use_mean
        self.use_cnt = use_cnt
        self.use_moments = use_moments
        self.use_skew_bias = use_skew_bias
        self.use_skew_nobias = use_skew_nobias
        self.use_kurtosis_fisher_bias = use_kurtosis_fisher_bias
        self.use_kurtosis_fisher_nobias = use_kurtosis_fisher_nobias
        self.use_kurtosis_pearson_bias = use_kurtosis_pearson_bias
        self.use_kurtosis_pearson_nobias = use_kurtosis_pearson_nobias
        self.use_quantile = use_quantile
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.n_jobs == 1:
            res = [self.transform_1(d) for d in data]
        else:
            def batch_job(ds):
                return [self.transform_1(d) for d in ds]
            res = Parallel(n_jobs=self.n_jobs)(delayed(batch_job)(d) for d in chunked_even(data, 32))
            res = list(itertools.chain.from_iterable(res))
        res = np.array(res)
        return res

    def transform_1(self, data_row):
        yes_data, no_data = data_row

        def get_stat_1(d):
            res = []
            if self.use_std:
                res.append(np.std(d))
            if self.use_var:
                res.append(np.var(d))
            if self.use_mean:
                res.append(np.mean(d))
            if self.use_cnt:
                res.append(len(d))
            for m in range(2, self.use_moments+1):
                res.append(stats.moment(d, m))
            if self.use_skew_bias:
                res.append(stats.skew(d, bias=True))
            if self.use_skew_nobias:
                res.append(stats.skew(d, bias=False))
            if self.use_kurtosis_fisher_bias:
                res.append(stats.kurtosis(d, fisher=True, bias=True))
            if self.use_kurtosis_fisher_nobias:
                res.append(stats.kurtosis(d, fisher=True, bias=False))
            if self.use_kurtosis_pearson_bias:
                res.append(stats.kurtosis(d, fisher=False, bias=True))
            if self.use_kurtosis_pearson_nobias:
                res.append(stats.kurtosis(d, fisher=False, bias=False))
            for m in np.linspace(0, 100, self.use_quantile):
                res.append(np.percentile(d, m))
            return res
        return np.array(list(itertools.chain.from_iterable((
            get_stat_1(yes_data),
            get_stat_1(no_data),
            get_stat_1([*yes_data, *no_data]),
        ))))
# dists = []
# for i in range(100):
#     d = [get_random_dist(), get_random_dist()]
#     dists.append([*d, get_score(d[0], d[1])])
#
# dists = sorted(dists, key=lambda x: x[2])
# fig, axes = plt.subplots(nrows=2, ncols=3)
# for i in range(3):
#     axes[0][i].scatter(np.random.random(len(dists[i][0])), dists[i][0], marker='o')
#     axes[0][i].scatter(np.random.random(len(dists[i][1]))+3, dists[i][1], marker='x')
#
# for i in range(3):
#     axes[1][i].scatter(np.random.random(len(dists[-1-i][0])), dists[-1-i][0], marker='o')
#     axes[1][i].scatter(np.random.random(len(dists[-1-i][1]))+3, dists[-1-i][1], marker='x')
# fig.show()
