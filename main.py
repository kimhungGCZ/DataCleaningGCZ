# coding: utf-8

# In[336]:

import matplotlib.pyplot as plt


# In[337]:

# Library for segment function
from numpy import arange, array, ones
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim, show
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# Library for SAX
import numpy as np
import pandas as pd
from fractions import Fraction
from functools import partial
from itertools import cycle
from joblib import Parallel, delayed
import joblib, tempfile, os
import cProfile


# In[338]:

# .....................SAX Function.....................

class SAXModel(object):
    def __init__(self, window=None, stride=None,
                 nbins=None, alphabet=None):
        """
        Assume a gapless (fixed freq. no missing value) time series
        window: sliding window length to define the number of words
        stride: stride of sliding, if stride < window, there is overlapping in windows
        nbins: number of bins in each sliding window, defining the length of word
        alphabet: alphabet for symbolization, also determines number of value levels
        Not all parameters are used if only partial functions of the class is needed
        """
        self.window = window
        self.stride = stride
        self.nbins = nbins
        self.alphabet = list(alphabet or "ABCD")
        self.nlevels = len(self.alphabet)

        if not (3 <= self.nlevels <= 20):
            raise ValueError("alphabet size is within 3 and 20 for current impl.")
        self.cutpoints = {3: [-np.inf, -0.43, 0.43, np.inf],
                          4: [-np.inf, -0.67, 0, 0.67, np.inf],
                          5: [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf],
                          6: [-np.inf, -0.97, -0.43, 0, 0.43, 0.97, np.inf],
                          7: [-np.inf, -1.07, -0.57, -0.18, 0.18, 0.57, 1.07, np.inf],
                          8: [-np.inf, -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, np.inf],
                          9: [-np.inf, -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, np.inf],
                          10: [-np.inf, -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28, np.inf],
                          11: [-np.inf, -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34, np.inf],
                          12: [-np.inf, -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38, np.inf],
                          13: [-np.inf, -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43,
                               np.inf],
                          14: [-np.inf, -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47,
                               np.inf],
                          15: [-np.inf, -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84,
                               1.11, 1.5, np.inf],
                          16: [-np.inf, -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67,
                               0.89, 1.15, 1.53, np.inf],
                          17: [-np.inf, -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54,
                               0.72, 0.93, 1.19, 1.56, np.inf],
                          18: [-np.inf, -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43,
                               0.59, 0.76, 0.97, 1.22, 1.59, np.inf],
                          19: [-np.inf, -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48,
                               0.63, 0.8, 1, 1.25, 1.62, np.inf],
                          20: [-np.inf, -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25,
                               0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, np.inf]}
        cps = self.cutpoints[len(self.alphabet)]
        vecs = map(lambda (a, b): (a + b) / 2, zip(cps, cps[1:]))  ## taking mean may not be accurate
        vecs[0] = cps[1]
        vecs[-1] = cps[-2]
        self.sym2vec = dict(zip(self.alphabet, vecs))

    def sliding_window_index(self, signal_length):
        """
        Takes length of signal and returns list of indices, each of which
        defines a sliding window
        """
        start = 0
        while (start + self.window) <= signal_length:
            yield slice(start, start + self.window)
            start += self.stride

    def whiten(self, window_signal):
        """
        Perform whitening - it should be local to a sliding window
        """
        s = np.asarray(window_signal)
        mu, sd = np.mean(s), np.std(s)
        return (s - mu) / (sd + 1e-10)

    def binpack(self, xs):
        """
        for a singal of length 5, nbins = 3,
        it generates (p1, 2*p2/3), (p2/3, p3, p4/3), (2*p4/3, p5)
        """
        xs = np.asarray(xs)
        binsize = Fraction(len(xs), self.nbins)
        wts = [1 for _ in xrange(int(binsize))] + [binsize - int(binsize)]
        pos = 0
        while pos < len(xs):
            n = len(wts) - 1 if wts[-1] == 0 else len(wts)
            yield xs[pos:(pos + n)] * wts[:n]
            pos += len(wts) - 1
            rest_wts = binsize - (1 - wts[-1])
            wts = [1 - wts[-1]] + [1 for _ in xrange(int(rest_wts))] + [rest_wts - int(rest_wts)]

    def optimize_binpack(self, xs):
        """
        for a singal of length 5, nbins = 3,
        it generates (p1, 2*p2/3), (p2/3, p3, p4/3), (2*p4/3, p5)
        """
        xs = np.asarray(xs)
        binsize = Fraction(len(xs), len(xs)/ (self.window/self.nbins))
        wts = [1 for _ in xrange(int(binsize))] + [binsize - int(binsize)]
        pos = 0
        while pos < len(xs):
            n = len(wts) - 1 if wts[-1] == 0 else len(wts)
            yield xs[pos:(pos + n)] * wts[:n]
            pos += len(wts) - 1
            rest_wts = binsize - (1 - wts[-1])
            wts = [1 - wts[-1]] + [1 for _ in xrange(int(rest_wts))] + [rest_wts - int(rest_wts)]

    def symbolize(self, xs):
        """
        Symbolize a PPA
        """
        alphabet_sz = len(self.alphabet)
        cutpoints = self.cutpoints[alphabet_sz]
        return pd.cut(xs, bins=cutpoints, labels=self.alphabet)

    def paa_window(self, window_signal):
        """
        piecewise aggregate approximation: one sliding window signal to a word
        """
        s = self.whiten(window_signal)
        binsize = Fraction(len(s), self.nbins)
        xs = map(lambda ss: np.sum(ss) / float(binsize), self.binpack(s))
        return xs

    def optimize_paa_window(self, window_signal):
        """
        piecewise aggregate approximation: one sliding window signal to a word
        """
        s = self.whiten(window_signal)
        size_bins = self.window / self.nbins
        binsize = Fraction(len(s), size_bins)
        xs = map(lambda ss: np.sum(ss) / float(binsize), self.optimize_binpack(s))
        return xs

    def optimize_symbolize_window(self, window_signal):
        """
        Symbolize one sliding window signal to a word
        """
        # s = self.whiten(window_signal)
        # binsize = Fraction(len(s), self.nbins)
        # xs = map(lambda ss: np.sum(ss) / float(binsize), self.binpack(s))
        xs = self.optimize_paa_window(window_signal)
        return "".join(self.symbolize(xs))

    def symbolize_window(self, window_signal):
        """
        Symbolize one sliding window signal to a word
        """
        # s = self.whiten(window_signal)
        # binsize = Fraction(len(s), self.nbins)
        # xs = map(lambda ss: np.sum(ss) / float(binsize), self.binpack(s))
        xs = self.paa_window(window_signal)
        return "".join(self.symbolize(xs))

    def symbolize_signal(self, signal, parallel=None, n_jobs=-1):
        """
        Symbolize whole time-series signal to a sentence (vector of words),
        parallel can be {None, "ipython"}
        """
        window_index = self.sliding_window_index(len(signal))
        if parallel == None:
            return map(lambda wi: self.symbolize_window(signal[wi]), window_index)
        elif parallel == "ipython":
            ## too slow
            raise NotImplementedError("parallel parameter %s not supported" % parallel)
        # return self.iparallel_symbolize_signal(signal)
        elif parallel == "joblib":
            with tempfile.NamedTemporaryFile(delete=False) as f:
                tf = f.name
            print "save temp file at %s" % tf
            tfiles = joblib.dump(signal, tf)
            xs = joblib.load(tf, "r")
            n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
            window_index = list(window_index)
            batch_size = len(window_index) / n_jobs
            batches = chunk(window_index, batch_size)
            symbols = Parallel(n_jobs)(delayed(joblib_symbolize_window)(self, xs, batch) for batch in batches)
            for f in tfiles: os.unlink(f)
            return sum(symbols, [])
        else:
            raise NotImplementedError("parallel parameter %s not supported" % parallel)

    def symbol_to_vector(self, words):
        return np.array([np.asarray([self.sym2vec[w] for w in word]) for word in words])

    def signal_to_paa_vector(self, signal, n_jobs=-1):
        window_index = self.sliding_window_index(len(signal))
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tf = f.name
        print "save temp file at %s" % tf
        tfiles = joblib.dump(signal, tf)
        xs = joblib.load(tf, "r")
        n_jobs = joblib.cpu_count() if n_jobs == -1 else n_jobs
        window_index = list(window_index)
        batch_size = len(window_index) / n_jobs
        batches = chunk(window_index, batch_size)
        vecs = Parallel(n_jobs)(delayed(joblib_paa_window)(self, xs, batch) for batch in batches)
        for f in tfiles: os.unlink(f)
        return np.vstack(vecs)

    def symbol_distance(self, word1, word2):
        cutpoints = self.cutpoints[len(self.alphabet)]
        inverted_alphabet = dict([(w, i) for (i, w) in enumerate(self.alphabet, 1)])
        diff = np.asarray([0 if abs(iw1 - iw2) <= 1 else cutpoints[max(iw1, iw2) - 1] - cutpoints[min(iw1, iw2)]
                           for (iw1, iw2) in zip(map(inverted_alphabet.get, word1), map(inverted_alphabet.get, word2))])
        return np.sqrt(np.sum(diff ** 2))

    def convert_index(self, word_indices=None, ts_indices=None):
        """
        if word_index is not None, convert word (sliding window) index to time series index
        otherwise convert ts_index to word_index
        """
        if word_indices is not None:
            return [wi * self.stride for wi in word_indices]
        elif ts_indices is not None:
            return [ti / self.stride for ti in ts_indices]
        else:
            raise ValueError("either word_index or ts_index needs to be specified")


## helper function
def joblib_symbolize_window(sax, xs, batch):
    return [sax.symbolize_window(xs[i]) for i in batch]


def joblib_paa_window(sax, xs, batch):
    return np.asarray([sax.paa_window(xs[i]) for i in batch])


def chunk(xs, chunk_size):
    p = 0
    while p < len(xs):
        yield xs[p:(p + chunk_size)]
        p += chunk_size


# In[339]:

def slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.
    The list is computed using the sliding window technique.
    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error
    """
    if not seq_range:
        seq_range = (0, len(sequence) - 1)

    start = seq_range[0]
    end = start
    result_segment = create_segment(sequence, (seq_range[0], seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment = create_segment(sequence, (start, end))
        error = compute_error(sequence, test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment, compute_error, max_error,
                                                       (end - 1, seq_range[1]))


def bottomupsegment(sequence, create_segment, compute_error, max_error):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the bottom-up technique.

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    segments = [create_segment(sequence, seq_range) for seq_range in
                zip(range(len(sequence))[:-1], range(len(sequence))[1:])]
    mergesegments = [create_segment(sequence, (seg1[0], seg2[2])) for seg1, seg2 in zip(segments[:-1], segments[1:])]
    mergecosts = [compute_error(sequence, segment) for segment in mergesegments]

    while min(mergecosts) < max_error:
        idx = mergecosts.index(min(mergecosts))
        segments[idx] = mergesegments[idx]
        del segments[idx + 1]

        if idx > 0:
            mergesegments[idx - 1] = create_segment(sequence, (segments[idx - 1][0], segments[idx][2]))
            mergecosts[idx - 1] = compute_error(sequence, mergesegments[idx - 1])

        if idx + 1 < len(mergecosts):
            mergesegments[idx + 1] = create_segment(sequence, (segments[idx][0], segments[idx + 1][2]))
            mergecosts[idx + 1] = compute_error(sequence, mergesegments[idx])

        del mergesegments[idx]
        del mergecosts[idx]

    return segments


def topdownsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the bottom-up technique.

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    if not seq_range:
        seq_range = (0, len(sequence) - 1)

    bestlefterror, bestleftsegment = float('inf'), None
    bestrighterror, bestrightsegment = float('inf'), None
    bestidx = None

    for idx in range(seq_range[0] + 1, seq_range[1]):
        segment_left = create_segment(sequence, (seq_range[0], idx))
        error_left = compute_error(sequence, segment_left)
        segment_right = create_segment(sequence, (idx, seq_range[1]))
        error_right = compute_error(sequence, segment_right)
        if error_left + error_right < bestlefterror + bestrighterror:
            bestlefterror, bestrighterror = error_left, error_right
            bestleftsegment, bestrightsegment = segment_left, segment_right
            bestidx = idx

    if bestlefterror <= max_error:
        leftsegs = [bestleftsegment]
    else:
        leftsegs = topdownsegment(sequence, create_segment, compute_error, max_error, (seq_range[0], bestidx))

    if bestrighterror <= max_error:
        rightsegs = [bestrightsegment]
    else:
        rightsegs = topdownsegment(sequence, create_segment, compute_error, max_error, (bestidx, seq_range[1]))

    return leftsegs + rightsegs


# In[340]:

def leastsquareslinefit(sequence, seq_range):
    """Return the parameters and error for a least squares line fit of one segment of a sequence"""
    x = arange(seq_range[0], seq_range[1] + 1)
    y = array(sequence[seq_range[0]:seq_range[1] + 1])
    A = ones((len(x), 2), float)
    A[:, 0] = x
    (p, residuals, rank, s) = lstsq(A, y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p, error)


def sumsquared_error(sequence, segment):
    """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""
    x0, y0, x1, y1 = segment
    p, error = leastsquareslinefit(sequence, (x0, x1))
    return error


# create_segment functions

def regression(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment of a sequence using linear regression"""
    p, error = leastsquareslinefit(sequence, seq_range)
    y0 = p[0] * seq_range[0] + p[1]
    y1 = p[0] * seq_range[1] + p[1]
    return (seq_range[0], y0, seq_range[1], y1)


def interpolate(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment using a simple interpolation"""
    return (seq_range[0], sequence[seq_range[0]], seq_range[1], sequence[seq_range[1]])


# In[341]:

def draw_plot(data, plot_title):
    plt.plot(range(len(data)), data, alpha=0.8, color='red')
    plt.title(plot_title)
    plt.xlabel("Samples")
    plt.ylabel("Signal")
    plt.xlim((0, len(data) - 1))
    plt.grid(True)


def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0], segment[2]), (segment[1], segment[3]))
        ax.add_line(line)


# In[342]:

''' IMPORT DATA 
from io import StringIO
import requests
import json
import pandas as pd


# @hidden_cell
# This function accesses a file in your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def get_object_storage_file_with_credentials_9436282b6e034b7e83ffd264c361b798(container, filename):
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
                                  'password': {'user': {'name': 'member_d6db630f3615c9ad3c060711ce137f12b1a5c29e',
                                                        'domain': {'id': '1e2abb72182d48229ea85b39cd9329b8'},
                                                        'password': 'fQht2S.V55HSGk*M'}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if (e1['type'] == 'object-store'):
            for e2 in e1['endpoints']:
                if (e2['interface'] == 'public' and e2['region'] == 'dallas'):
                    url2 = ''.join([e2['url'], '/', container, '/', filename])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO(resp2.text)


df_data_1 = pd.read_csv(
    get_object_storage_file_with_credentials_9436282b6e034b7e83ffd264c361b798('SAXtesting', 'data.csv'))
print(df_data_1.info())
print(len(df_data_1.water_temperature))
'''
# In[343]:

data = [19.67860335,21.86011173,22.2450838,20.57687151,19.42195531,20.19189944,18.65201117,20.57687151,19.80692737,20.96184358,19.9352514,20.83351955,20.19189944,21.34681564,21.86011173,20.96184358,22.63005587,21.60346369,22.63005587,22.63005587,22.75837989,23.01502793,22.88670391,23.14335196,23.27167598,24.04162011,24.29826816,25.19653631,25.06821229,26.35145251,27.37804469,28.91793296,31.86938547,13.00575419,12.74910615,13.51905028,12.87743017,13.00575419,13.13407821,13.90402235,13.90402235,14.54564246,14.54564246,15.31558659,15.05893855,15.57223464,15.18726257,15.70055866,15.44391061,15.70055866,15.9572067,16.34217877,16.21385475,16.59882682,16.72715084,16.98379888,16.85547486,17.49709497,17.62541899,18.01039106,18.26703911,18.7803352,18.65201117,19.29363128,19.55027933,19.80692737,20.57687151,20.44854749,20.70519553,20.57687151,20.32022346,19.42195531,19.9352514,19.80692737,20.57687151,20.44854749,20.57687151,20.70519553,21.21849162,21.0901676,21.60346369,21.60346369,22.2450838,21.98843575,22.75837989,22.75837989,23.14335196,23.14335196,23.65664804,23.52832402,24.29826816,24.16994413,24.93988827,24.68324022,25.58150838,25.7098324,26.8647486,26.99307263,27.63469274,27.89134078,28.53296089,28.91793296,29.68787709,30.32949721,31.35608939,33.92256983,31.35608939,33.40927374,31.09944134,26.35145251,17.24044693,17.24044693,17.62541899,17.62541899,17.75374302,15.9572067,16.34217877,16.34217877,16.72715084,16.21385475,16.47050279,16.72715084,17.11212291,16.98379888,17.75374302,17.49709497,18.01039106,17.88206704,18.13871508,18.26703911,18.39536313,18.52368715,18.65201117,18.7803352,19.16530726,19.29363128,19.67860335,19.55027933,19.55027933,20.06357542,19.67860335,20.32022346,20.32022346,20.44854749,20.32022346,20.57687151,20.83351955,21.0901676,21.34681564,21.73178771,22.11675978,22.63005587,22.88670391,23.14335196,23.14335196,23.27167598,23.91329609,24.16994413,24.42659218,24.29826816,24.29826816,23.4,18.39536313,18.52368715,18.26703911,18.52368715,18.26703911,17.24044693,16.59882682,15.70055866,13.00575419,12.62078212,12.10748603,12.23581006,12.23581006,12.10748603,12.4924581,12.4924581,12.4924581,12.36413408,12.4924581,12.62078212,12.74910615,12.74910615,13.00575419,13.00575419,13.26240223,13.39072626,13.51905028,13.6473743,13.77569832,13.51905028,13.51905028,13.51905028,13.39072626,13.51905028,13.77569832,13.39072626,13.51905028,13.51905028,13.6473743,13.77569832,13.6473743,13.77569832,13.77569832,13.6473743,13.51905028,13.77569832,13.51905028,13.51905028,13.6473743,13.6473743,14.03234637,13.77569832,13.77569832,13.51905028,13.77569832,13.51905028,13.6473743,13.77569832,13.77569832,13.77569832,13.77569832,13.26240223,13.51905028,13.51905028,13.77569832,13.51905028,13.77569832,13.90402235,13.77569832,13.90402235,14.16067039,14.67396648,14.8022905,14.54564246,14.41731844,14.28899441,14.67396648,14.54564246,14.67396648,14.8022905,14.67396648,14.93061453,15.18726257,15.31558659,15.57223464,15.82888268,15.82888268,16.08553073,15.70055866,15.05893855,10.82424581,10.69592179,11.08089385,10.95256983,10.95256983,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,10.82424581,10.69592179,10.82424581,10.95256983,11.08089385,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,11.08089385,10.82424581,11.08089385,11.08089385,11.08089385,10.56759777,11.3375419,11.20921788,11.46586592,11.20921788,11.46586592,11.3375419,11.72251397,11.08089385,12.10748603,11.59418994,12.36413408,11.97916201,11.97916201,11.72251397,11.85083799,11.97916201,12.23581006,12.74910615,13.51905028,13.6473743,13.90402235,13.51905028,13.26240223,13.13407821,13.00575419,13.26240223,13.39072626,13.77569832,13.90402235,13.6473743,13.39072626,13.77569832,13.51905028,13.77569832,14.03234637,14.16067039,14.28899441,14.41731844,14.41731844,14.8022905,14.8022905,15.05893855,15.70055866,15.82888268,15.31558659,15.57223464,15.70055866,15.82888268,15.82888268,15.70055866,16.21385475,16.34217877,16.47050279,16.21385475,18.52368715,18.65201117,19.29363128,19.03698324,19.29363128,18.26703911,18.7803352,18.39536313,17.88206704,17.36877095,17.49709497,17.36877095,17.75374302,18.26703911,19.16530726,19.9352514,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,8.001117318,8.129441341,8.129441341,8.129441341,7.872793296,8.001117318,7.872793296,7.872793296,7.744469274,8.001117318,7.872793296,7.872793296,7.744469274,7.872793296,7.872793296,8.129441341,7.744469274,8.001117318,7.744469274,8.001117318,7.616145251,7.616145251,7.359497207,8.129441341,7.872793296,8.129441341,8.129441341,8.386089385,8.386089385,8.386089385,7.359497207,0.045027933,0.173351955,17.62541899,17.75374302,17.36877095,16.98379888,17.88206704,17.49709497,18.39536313,17.62541899,18.39536313,18.13871508,18.39536313,18.39536313,18.39536313,18.26703911,18.26703911,18.39536313,18.26703911,18.26703911,18.13871508,18.26703911,18.52368715,18.65201117,18.52368715,18.52368715,18.65201117,18.65201117,18.39536313,18.26703911,18.39536313,18.52368715,19.03698324,19.03698324,19.03698324,19.16530726,19.16530726,18.90865922,18.90865922,18.65201117,18.90865922,19.03698324,19.42195531,19.67860335,19.80692737,19.80692737,19.80692737,20.32022346,20.19189944,20.19189944,20.19189944,20.32022346,21.0901676,21.47513966,21.21849162,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.21849162,21.34681564,21.98843575,22.11675978,22.11675978,22.11675978,21.98843575,22.11675978,21.98843575,21.98843575,22.11675978,22.37340782,22.37340782,22.63005587,22.88670391,22.88670391,22.88670391,22.75837989,22.75837989,22.75837989,22.88670391,23.01502793,23.78497207,24.16994413,24.04162011,24.04162011,24.04162011,23.91329609,23.91329609,23.78497207,24.16994413,24.16994413,24.81156425,24.93988827,25.06821229,25.06821229,25.06821229,24.93988827,24.81156425,25.06821229,24.93988827,25.19653631,25.83815642,26.09480447,26.09480447,26.09480447,25.96648045,25.96648045,25.96648045,25.83815642,25.83815642,25.96648045,26.47977654,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,26.60810056,26.60810056,26.8647486,26.73642458,27.24972067,27.24972067,27.50636872,27.37804469,27.37804469,27.37804469,27.24972067,27.24972067,27.24972067,27.37804469,27.63469274,27.89134078,28.0196648,27.63469274,27.76301676,27.89134078,27.63469274,27.50636872,27.50636872,27.76301676,28.0196648,28.14798883,28.27631285,28.27631285,28.0196648,28.14798883,28.14798883,28.14798883,28.27631285,28.27631285,28.40463687,28.40463687,28.53296089,28.66128492,28.66128492,28.66128492,28.40463687,28.27631285,28.27631285,28.40463687,28.91793296,28.91793296,28.91793296,29.04625698,29.17458101,29.30290503,29.30290503,29.30290503,29.43122905,29.43122905,29.81620112,29.55955307,29.55955307,29.17458101,29.43122905,28.14798883,28.14798883,28.0196648,28.0196648,28.14798883,28.78960894,29.04625698,29.30290503,29.17458101,29.17458101,29.30290503,29.43122905,29.43122905,29.68787709,29.94452514,30.71446927,30.97111732,31.22776536,31.35608939,31.48441341,31.9977095,32.25435754,32.51100559,33.28094972,34.43586592,20.70519553,20.96184358,20.96184358,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.21849162,21.21849162,21.34681564,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.60346369,21.60346369,21.86011173,21.86011173,21.86011173,21.86011173,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.11675978,22.2450838,22.2450838,22.2450838,22.37340782,22.37340782,22.63005587,22.63005587,23.27167598,23.27167598,23.52832402,23.52832402,23.65664804,23.52832402,23.52832402,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.78497207,23.78497207,24.04162011,24.04162011,24.81156425,25.19653631,25.19653631,25.06821229,25.06821229,25.19653631,25.19653631,25.06821229,25.06821229,25.06821229,25.06821229,25.19653631,25.19653631,24.93988827,24.93988827,25.06821229,25.06821229,25.58150838,25.58150838,26.09480447,26.35145251,26.47977654,26.35145251,26.47977654,26.35145251,26.35145251,26.35145251,26.22312849,26.47977654,26.60810056,28.27631285,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,28.14798883,27.76301676,27.76301676,28.0196648,28.27631285,28.27631285,28.14798883,28.0196648,28.14798883,28.14798883,28.27631285,28.14798883,28.14798883,28.27631285,28.53296089,28.53296089,28.40463687,28.40463687,28.53296089,28.53296089,28.53296089,28.40463687,28.53296089,28.78960894,29.17458101,29.17458101,29.04625698,29.17458101,29.17458101,29.17458101,29.17458101,29.43122905,29.81620112,30.20117318,30.58614525,30.71446927,30.8427933,30.8427933,31.35608939,31.48441341,31.74106145,32.25435754,32.89597765,35.97575419,23.4,23.52832402,23.52832402,23.52832402,23.27167598,23.4,23.52832402,23.52832402,23.91329609,24.5549162,24.68324022,24.68324022,24.81156425,24.68324022,24.68324022,24.81156425,24.81156425,24.81156425,25.32486034,25.83815642,26.09480447,26.22312849,26.22312849,26.22312849,26.22312849,26.09480447,26.09480447,26.22312849,26.47977654,26.99307263,27.12139665,27.37804469,27.24972067,27.24972067,27.24972067,27.24972067,27.24972067,27.37804469,27.37804469,27.63469274,27.76301676,27.89134078,27.89134078,27.89134078,28.14798883,28.0196648,28.14798883,28.14798883,28.40463687,29.17458101,29.17458101,29.55955307,29.43122905,29.55955307,29.43122905,29.43122905,29.30290503,29.55955307,29.81620112,30.58614525,30.71446927,30.71446927,31.09944134,31.22776536,31.48441341,31.61273743,32.12603352,32.51100559,33.66592179,19.16530726,19.55027933,19.55027933,19.55027933,19.67860335,19.80692737,19.9352514,19.9352514,19.9352514,20.06357542,20.19189944,20.57687151,20.83351955,20.83351955,20.96184358,21.0901676,21.0901676,21.34681564,21.47513966,21.47513966,21.73178771,22.11675978,22.2450838,22.2450838,22.37340782,22.37340782,22.2450838,22.37340782,22.50173184,22.63005587,23.65664804,23.91329609,23.91329609,23.91329609,23.91329609,23.78497207,23.65664804,23.78497207,24.16994413,24.29826816,25.06821229,25.19653631,25.19653631,25.19653631,25.19653631,25.32486034,25.19653631,25.06821229,25.19653631,25.45318436,25.7098324,25.83815642,25.96648045,25.83815642,25.83815642,25.7098324,25.83815642,25.7098324,25.7098324,26.09480447,26.73642458,27.12139665,27.12139665,27.12139665,26.99307263,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,27.37804469,27.50636872,27.50636872,27.76301676,27.76301676,27.63469274,27.76301676,27.76301676,27.63469274,27.63469274,27.76301676,27.76301676,27.89134078,27.76301676,27.63469274,27.63469274,27.63469274,27.76301676,27.89134078,28.27631285,28.78960894,29.17458101,29.55955307,29.55955307,29.68787709,29.94452514,30.32949721,30.71446927,31.35608939,32.51100559,17.24044693,17.49709497,17.62541899,17.75374302,17.62541899,17.88206704,18.01039106,17.88206704,17.88206704,17.88206704,18.01039106,18.26703911,18.39536313,18.39536313,18.39536313,18.52368715,18.65201117,18.7803352,18.52368715,18.90865922,18.7803352,18.90865922,19.03698324,19.16530726,19.03698324,19.29363128,19.29363128,19.42195531,19.29363128,19.29363128,19.55027933,19.67860335,19.9352514,19.80692737,19.9352514,20.06357542,19.9352514,20.19189944,20.19189944,20.19189944,20.44854749,20.32022346,20.57687151,20.44854749,20.44854749,20.57687151,20.57687151,20.57687151,20.57687151,20.83351955,20.96184358,21.0901676,21.34681564,21.34681564,21.34681564,21.21849162,21.0901676,21.21849162,21.21849162,21.47513966,21.60346369,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.2450838,22.88670391,22.88670391,22.88670391,23.01502793,22.88670391,22.88670391,22.88670391,22.88670391,23.01502793,23.01502793,23.4,23.4,23.78497207,23.78497207,23.78497207,23.4,23.4,23.4,23.52832402,23.78497207,24.29826816,24.5549162,24.42659218,24.5549162,24.42659218,24.29826816,24.29826816,24.16994413,24.16994413,24.29826816,24.81156425,24.93988827,24.93988827,24.93988827,24.93988827,24.68324022,24.68324022,24.68324022,24.68324022,24.81156425,25.06821229,25.32486034,25.45318436,25.58150838,25.45318436,25.58150838,25.45318436,25.32486034,25.45318436,25.7098324,25.96648045,26.09480447,26.35145251,26.35145251,26.60810056,26.60810056,26.8647486,26.8647486,27.12139665,27.37804469,27.89134078,31.09944134,21.60346369,21.86011173,22.11675978,22.50173184,22.88670391,23.01502793,23.52832402,23.52832402,24.81156425,27.24972067,34.56418994,48.87527933,5.947932961]
#data = df_data_1.water_temperature
data = data[0:200]
max_error = 0.5

ys2 = range(len(data))
filtered = lowess(data, ys2, is_sorted=True, frac=0.01, it=0)
nd_filtered = map(lambda x: x[1], filtered)

FFT_data = np.fft.fft(nd_filtered);
freq = np.fft.fftfreq(200)

#print FFT_data
plt.figure(1, figsize=(28, 12), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(211)
plt.title("Smooth data")
plt.plot(data)
plt.subplot(212)
plt.plot(freq, 'r--')
plt.show()
''''
# print nd_filtered
'''
plt.figure(2, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(211)
segments = slidingwindowsegment(nd_filtered, regression, sumsquared_error, max_error)
draw_plot(data,"Sliding window with regression")
draw_segments(segments)
############################################################
plt.figure(3, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(211)
segments = bottomupsegment(nd_filtered, regression, sumsquared_error, max_error)
draw_plot(data,"Bottom-up with regression")
draw_segments(segments)
'''
##############################################################
plt.figure(4, figsize=(28, 12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(211)

print "Smothed data Length: ", len(nd_filtered)

segments = slidingwindowsegment(nd_filtered, interpolate, sumsquared_error, max_error)
draw_plot(data, "Sliding window with simple interpolation")
draw_segments(segments)

print "Segment Length: ", len(segments)

new_segments1 = map(lambda x: [x[0], x[1]], segments)
new_segments1.append(list([segments[len(segments) - 1][2], segments[len(segments) - 1][3]]))


def percentatge_diff(a, b):
    return abs((a - b) / ((a + b) / 2) * 100)


def check_beakpoint(point, i, data, pf_threshold):
    if (i > 1 and i < len(data) - 2):
        return (percentatge_diff(point[1], data[i - 1][1][1]) > pf_threshold) and (point[1]) <= min(data[i - 2][1][1],
                                                                                                    data[i - 1][1][1],
                                                                                                    point[1],
                                                                                                    data[i + 1][1][1],
                                                                                                    data[i + 2][1][1])
    else:
        return False


less_than_pf_threshold = list(
    filter(lambda (i, x): check_beakpoint(x, i, list(enumerate(new_segments1)), 5), enumerate(new_segments1)))
less_than_pf_threshold = map(lambda x: x[1], less_than_pf_threshold)
less_than_pf_threshold = list(less_than_pf_threshold)
print "The number of Strange Point: ", len(less_than_pf_threshold)
plt.plot(map(lambda x: x[0], less_than_pf_threshold), map(lambda x: x[1], less_than_pf_threshold), 'ro', color='yellow',
         markersize=15)

plt.show()
# show()
'''

# In[344]:

''' __________Test SAX Model____________ 

sax = SAXModel(nbins=5, alphabet="ABCD")
sax.sym2vec

# In[345]:

print less_than_pf_threshold
window = 50
stride = window
nbins = 5
alphabet = "ABCDEFGHIJKLMN"
# xs1 = [19.67860335,21.86011173,22.2450838,20.57687151,19.42195531,20.19189944,18.65201117,20.57687151,19.80692737,20.96184358,19.9352514,20.83351955,20.19189944,21.34681564,21.86011173,20.96184358,22.63005587,21.60346369,22.63005587,22.63005587,22.75837989,23.01502793,22.88670391,23.14335196,23.27167598,24.04162011,24.29826816,25.19653631,25.06821229,26.35145251,27.37804469,28.91793296,31.86938547,13.00575419,12.74910615,13.51905028,12.87743017,13.00575419,13.13407821,13.90402235,13.90402235,14.54564246,14.54564246,15.31558659,15.05893855,15.57223464,15.18726257,15.70055866,15.44391061,15.70055866,15.9572067,16.34217877,16.21385475,16.59882682,16.72715084,16.98379888,16.85547486,17.49709497,17.62541899,18.01039106,18.26703911,18.7803352,18.65201117,19.29363128,19.55027933,19.80692737,20.57687151,20.44854749,20.70519553,20.57687151,20.32022346,19.42195531,19.9352514,19.80692737,20.57687151,20.44854749,20.57687151,20.70519553,21.21849162,21.0901676,21.60346369,21.60346369,22.2450838,21.98843575,22.75837989,22.75837989,23.14335196,23.14335196,23.65664804,23.52832402,24.29826816,24.16994413,24.93988827,24.68324022,25.58150838,25.7098324,26.8647486,26.99307263,27.63469274,27.89134078,28.53296089,28.91793296,29.68787709,30.32949721,31.35608939,33.92256983,31.35608939,33.40927374,31.09944134,26.35145251,17.24044693,17.24044693,17.62541899,17.62541899,17.75374302,15.9572067,16.34217877,16.34217877,16.72715084,16.21385475,16.47050279,16.72715084,17.11212291,16.98379888,17.75374302,17.49709497,18.01039106,17.88206704,18.13871508,18.26703911,18.39536313,18.52368715,18.65201117,18.7803352,19.16530726,19.29363128,19.67860335,19.55027933,19.55027933,20.06357542,19.67860335,20.32022346,20.32022346,20.44854749,20.32022346,20.57687151,20.83351955,21.0901676,21.34681564,21.73178771,22.11675978,22.63005587,22.88670391,23.14335196,23.14335196,23.27167598,23.91329609,24.16994413,24.42659218,24.29826816,24.29826816,23.4,18.39536313,18.52368715,18.26703911,18.52368715,18.26703911,17.24044693,16.59882682,15.70055866,13.00575419,12.62078212,12.10748603,12.23581006,12.23581006,12.10748603,12.4924581,12.4924581,12.4924581,12.36413408,12.4924581,12.62078212,12.74910615,12.74910615,13.00575419,13.00575419,13.26240223,13.39072626,13.51905028,13.6473743,13.77569832,13.51905028,13.51905028,13.51905028,13.39072626,13.51905028,13.77569832,13.39072626,13.51905028,13.51905028,13.6473743,13.77569832,13.6473743,13.77569832,13.77569832,13.6473743,13.51905028,13.77569832,13.51905028,13.51905028,13.6473743,13.6473743,14.03234637,13.77569832,13.77569832,13.51905028,13.77569832,13.51905028,13.6473743,13.77569832,13.77569832,13.77569832,13.77569832,13.26240223,13.51905028,13.51905028,13.77569832,13.51905028,13.77569832,13.90402235,13.77569832,13.90402235,14.16067039,14.67396648,14.8022905,14.54564246,14.41731844,14.28899441,14.67396648,14.54564246,14.67396648,14.8022905,14.67396648,14.93061453,15.18726257,15.31558659,15.57223464,15.82888268,15.82888268,16.08553073,15.70055866,15.05893855,10.82424581,10.69592179,11.08089385,10.95256983,10.95256983,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,10.82424581,10.69592179,10.82424581,10.95256983,11.08089385,10.95256983,11.08089385,10.95256983,10.95256983,10.82424581,11.08089385,10.82424581,11.08089385,11.08089385,11.08089385,10.56759777,11.3375419,11.20921788,11.46586592,11.20921788,11.46586592,11.3375419,11.72251397,11.08089385,12.10748603,11.59418994,12.36413408,11.97916201,11.97916201,11.72251397,11.85083799,11.97916201,12.23581006,12.74910615,13.51905028,13.6473743,13.90402235,13.51905028,13.26240223,13.13407821,13.00575419,13.26240223,13.39072626,13.77569832,13.90402235,13.6473743,13.39072626,13.77569832,13.51905028,13.77569832,14.03234637,14.16067039,14.28899441,14.41731844,14.41731844,14.8022905,14.8022905,15.05893855,15.70055866,15.82888268,15.31558659,15.57223464,15.70055866,15.82888268,15.82888268,15.70055866,16.21385475,16.34217877,16.47050279,16.21385475,18.52368715,18.65201117,19.29363128,19.03698324,19.29363128,18.26703911,18.7803352,18.39536313,17.88206704,17.36877095,17.49709497,17.36877095,17.75374302,18.26703911,19.16530726,19.9352514,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,8.001117318,8.129441341,8.129441341,8.129441341,7.872793296,8.001117318,7.872793296,7.872793296,7.744469274,8.001117318,7.872793296,7.872793296,7.744469274,7.872793296,7.872793296,8.129441341,7.744469274,8.001117318,7.744469274,8.001117318,7.616145251,7.616145251,7.359497207,8.129441341,7.872793296,8.129441341,8.129441341,8.386089385,8.386089385,8.386089385,7.359497207,0.045027933,0.173351955,17.62541899,17.75374302,17.36877095,16.98379888,17.88206704,17.49709497,18.39536313,17.62541899,18.39536313,18.13871508,18.39536313,18.39536313,18.39536313,18.26703911,18.26703911,18.39536313,18.26703911,18.26703911,18.13871508,18.26703911,18.52368715,18.65201117,18.52368715,18.52368715,18.65201117,18.65201117,18.39536313,18.26703911,18.39536313,18.52368715,19.03698324,19.03698324,19.03698324,19.16530726,19.16530726,18.90865922,18.90865922,18.65201117,18.90865922,19.03698324,19.42195531,19.67860335,19.80692737,19.80692737,19.80692737,20.32022346,20.19189944,20.19189944,20.19189944,20.32022346,21.0901676,21.47513966,21.21849162,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.21849162,21.34681564,21.98843575,22.11675978,22.11675978,22.11675978,21.98843575,22.11675978,21.98843575,21.98843575,22.11675978,22.37340782,22.37340782,22.63005587,22.88670391,22.88670391,22.88670391,22.75837989,22.75837989,22.75837989,22.88670391,23.01502793,23.78497207,24.16994413,24.04162011,24.04162011,24.04162011,23.91329609,23.91329609,23.78497207,24.16994413,24.16994413,24.81156425,24.93988827,25.06821229,25.06821229,25.06821229,24.93988827,24.81156425,25.06821229,24.93988827,25.19653631,25.83815642,26.09480447,26.09480447,26.09480447,25.96648045,25.96648045,25.96648045,25.83815642,25.83815642,25.96648045,26.47977654,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,26.60810056,26.60810056,26.8647486,26.73642458,27.24972067,27.24972067,27.50636872,27.37804469,27.37804469,27.37804469,27.24972067,27.24972067,27.24972067,27.37804469,27.63469274,27.89134078,28.0196648,27.63469274,27.76301676,27.89134078,27.63469274,27.50636872,27.50636872,27.76301676,28.0196648,28.14798883,28.27631285,28.27631285,28.0196648,28.14798883,28.14798883,28.14798883,28.27631285,28.27631285,28.40463687,28.40463687,28.53296089,28.66128492,28.66128492,28.66128492,28.40463687,28.27631285,28.27631285,28.40463687,28.91793296,28.91793296,28.91793296,29.04625698,29.17458101,29.30290503,29.30290503,29.30290503,29.43122905,29.43122905,29.81620112,29.55955307,29.55955307,29.17458101,29.43122905,28.14798883,28.14798883,28.0196648,28.0196648,28.14798883,28.78960894,29.04625698,29.30290503,29.17458101,29.17458101,29.30290503,29.43122905,29.43122905,29.68787709,29.94452514,30.71446927,30.97111732,31.22776536,31.35608939,31.48441341,31.9977095,32.25435754,32.51100559,33.28094972,34.43586592,20.70519553,20.96184358,20.96184358,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.0901676,21.21849162,21.21849162,21.34681564,21.34681564,21.34681564,21.34681564,21.21849162,21.21849162,21.60346369,21.60346369,21.86011173,21.86011173,21.86011173,21.86011173,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.11675978,22.2450838,22.2450838,22.2450838,22.37340782,22.37340782,22.63005587,22.63005587,23.27167598,23.27167598,23.52832402,23.52832402,23.65664804,23.52832402,23.52832402,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.65664804,23.78497207,23.78497207,24.04162011,24.04162011,24.81156425,25.19653631,25.19653631,25.06821229,25.06821229,25.19653631,25.19653631,25.06821229,25.06821229,25.06821229,25.06821229,25.19653631,25.19653631,24.93988827,24.93988827,25.06821229,25.06821229,25.58150838,25.58150838,26.09480447,26.35145251,26.47977654,26.35145251,26.47977654,26.35145251,26.35145251,26.35145251,26.22312849,26.47977654,26.60810056,28.27631285,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,27.63469274,28.14798883,27.76301676,27.76301676,28.0196648,28.27631285,28.27631285,28.14798883,28.0196648,28.14798883,28.14798883,28.27631285,28.14798883,28.14798883,28.27631285,28.53296089,28.53296089,28.40463687,28.40463687,28.53296089,28.53296089,28.53296089,28.40463687,28.53296089,28.78960894,29.17458101,29.17458101,29.04625698,29.17458101,29.17458101,29.17458101,29.17458101,29.43122905,29.81620112,30.20117318,30.58614525,30.71446927,30.8427933,30.8427933,31.35608939,31.48441341,31.74106145,32.25435754,32.89597765,35.97575419,23.4,23.52832402,23.52832402,23.52832402,23.27167598,23.4,23.52832402,23.52832402,23.91329609,24.5549162,24.68324022,24.68324022,24.81156425,24.68324022,24.68324022,24.81156425,24.81156425,24.81156425,25.32486034,25.83815642,26.09480447,26.22312849,26.22312849,26.22312849,26.22312849,26.09480447,26.09480447,26.22312849,26.47977654,26.99307263,27.12139665,27.37804469,27.24972067,27.24972067,27.24972067,27.24972067,27.24972067,27.37804469,27.37804469,27.63469274,27.76301676,27.89134078,27.89134078,27.89134078,28.14798883,28.0196648,28.14798883,28.14798883,28.40463687,29.17458101,29.17458101,29.55955307,29.43122905,29.55955307,29.43122905,29.43122905,29.30290503,29.55955307,29.81620112,30.58614525,30.71446927,30.71446927,31.09944134,31.22776536,31.48441341,31.61273743,32.12603352,32.51100559,33.66592179,19.16530726,19.55027933,19.55027933,19.55027933,19.67860335,19.80692737,19.9352514,19.9352514,19.9352514,20.06357542,20.19189944,20.57687151,20.83351955,20.83351955,20.96184358,21.0901676,21.0901676,21.34681564,21.47513966,21.47513966,21.73178771,22.11675978,22.2450838,22.2450838,22.37340782,22.37340782,22.2450838,22.37340782,22.50173184,22.63005587,23.65664804,23.91329609,23.91329609,23.91329609,23.91329609,23.78497207,23.65664804,23.78497207,24.16994413,24.29826816,25.06821229,25.19653631,25.19653631,25.19653631,25.19653631,25.32486034,25.19653631,25.06821229,25.19653631,25.45318436,25.7098324,25.83815642,25.96648045,25.83815642,25.83815642,25.7098324,25.83815642,25.7098324,25.7098324,26.09480447,26.73642458,27.12139665,27.12139665,27.12139665,26.99307263,26.8647486,26.8647486,26.8647486,26.8647486,26.8647486,27.37804469,27.50636872,27.50636872,27.76301676,27.76301676,27.63469274,27.76301676,27.76301676,27.63469274,27.63469274,27.76301676,27.76301676,27.89134078,27.76301676,27.63469274,27.63469274,27.63469274,27.76301676,27.89134078,28.27631285,28.78960894,29.17458101,29.55955307,29.55955307,29.68787709,29.94452514,30.32949721,30.71446927,31.35608939,32.51100559,17.24044693,17.49709497,17.62541899,17.75374302,17.62541899,17.88206704,18.01039106,17.88206704,17.88206704,17.88206704,18.01039106,18.26703911,18.39536313,18.39536313,18.39536313,18.52368715,18.65201117,18.7803352,18.52368715,18.90865922,18.7803352,18.90865922,19.03698324,19.16530726,19.03698324,19.29363128,19.29363128,19.42195531,19.29363128,19.29363128,19.55027933,19.67860335,19.9352514,19.80692737,19.9352514,20.06357542,19.9352514,20.19189944,20.19189944,20.19189944,20.44854749,20.32022346,20.57687151,20.44854749,20.44854749,20.57687151,20.57687151,20.57687151,20.57687151,20.83351955,20.96184358,21.0901676,21.34681564,21.34681564,21.34681564,21.21849162,21.0901676,21.21849162,21.21849162,21.47513966,21.60346369,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,21.98843575,22.11675978,22.2450838,22.88670391,22.88670391,22.88670391,23.01502793,22.88670391,22.88670391,22.88670391,22.88670391,23.01502793,23.01502793,23.4,23.4,23.78497207,23.78497207,23.78497207,23.4,23.4,23.4,23.52832402,23.78497207,24.29826816,24.5549162,24.42659218,24.5549162,24.42659218,24.29826816,24.29826816,24.16994413,24.16994413,24.29826816,24.81156425,24.93988827,24.93988827,24.93988827,24.93988827,24.68324022,24.68324022,24.68324022,24.68324022,24.81156425,25.06821229,25.32486034,25.45318436,25.58150838,25.45318436,25.58150838,25.45318436,25.32486034,25.45318436,25.7098324,25.96648045,26.09480447,26.35145251,26.35145251,26.60810056,26.60810056,26.8647486,26.8647486,27.12139665,27.37804469,27.89134078,31.09944134,21.60346369,21.86011173,22.11675978,22.50173184,22.88670391,23.01502793,23.52832402,23.52832402,24.81156425,27.24972067,34.56418994,48.87527933,5.947932961,]
# xs = xs1[0:800]
xs = data
SAX_result = []

# xs = 2.5 * np.random.randn(100) + 3
sax = SAXModel(window, stride, nbins, alphabet)

plt.figure(1, figsize=(48, 12), dpi=80, facecolor='w', edgecolor='k')
plt.title('Optimized SAX result: \n')
##### Calculating window size from breakpoints:
windowsList = []
for i, breakpoint in enumerate(less_than_pf_threshold):
    if i == 0:
        windowsList.append(slice(i, breakpoint[0]))
        windowsList.append(slice(breakpoint[0], less_than_pf_threshold[i + 1][0]))
    elif i < len(less_than_pf_threshold) - 1:
        windowsList.append(slice(breakpoint[0], less_than_pf_threshold[i + 1][0]))
    else:
        windowsList.append(slice(breakpoint[0], len(xs)))

##### Add the calculated windows size to SAX model
# window = list(sax.sliding_window_index(len(xs)))
window = windowsList

for index, row in enumerate(window):
    if index == 0:
        pos_coll_main = 0
    else:
        pos_coll_main = less_than_pf_threshold[index - 1][0]
    plt.axvline(x=pos_coll_main, linewidth=2, color='#000000')
    ppa_array = sax.optimize_paa_window(xs[row])
    sax_result = sax.optimize_symbolize_window(xs[row])
    ss = sax.whiten(xs[row])
    # print "PPA Value: ", ppa_array
    SAX_result.append((sax_result));
    plt.plot(range(0, len(xs))[row], ss)
    for index1, row1 in enumerate(ppa_array):
        pos_coll = pos_coll_main + index1 * len(xs[row]) / nbins
        pos_coll_nex = pos_coll_main + (index1 + 1) * len(xs[row]) / nbins
        if pos_coll == pos_coll_main:
            plt.axvline(x=pos_coll, linewidth=2, color='#000000', linestyle='dashdot')
            # else:
            # plt.axvline(x = pos_coll, linewidth=2, color='#d62728',linestyle='dashdot')
        plt.plot((pos_coll + pos_coll_nex) / 2, row1, 'bo', color='green', markersize=10)

letter_array1 = sax.cutpoints[len(alphabet)]
letter_array = filter(lambda x: np.isinf(x) != True, letter_array1)
# print letter_array
j = 0
z = 0
for i in letter_array:
    plt.axhline(y=i, linewidth=2, color='#1f77b4', linestyle='dashdot')
    plt.text(0.2, i - 0.3, alphabet[j], fontsize=15, color='#d62728')
    j = j + 1
    z = i
plt.text(0.2, z + 0.3, alphabet[j], fontsize=15, color='#d62728')
# plt.grid(True)
# plt.plot(ss)
plt.show
print "Optimized SAX result: \n", SAX_result
Vector_result = map(lambda x: sax.symbol_to_vector(x), SAX_result)

##############################################################################################################################################################
############################################# NORMAL CASE ###################################################################################################
############################################################################################################################################################

normal_sax_result = sax.symbolize_signal(xs)
print "Normal SAX result: \n", normal_sax_result

plt.figure(2, figsize=(48, 8), dpi=80, facecolor='w', edgecolor='k')
plt.title('Normal SAX result: \n')
##### Calculating window size from breakpoints:
windowsList = []
for i in range(len(normal_sax_result)):
    windowsList.append(slice(i * 50, (i + 1) * 50))

##### Add the calculated windows size to SAX model
# window = list(sax.sliding_window_index(len(xs)))
window = windowsList
normal_ppa_result = map(lambda x: sax.paa_window(xs[x]), window)

for index, row in enumerate(window):
    pos_coll_main = min(range(0, len(xs))[row])
    plt.axvline(x=pos_coll_main, linewidth=2, color='#000000')
    ss = sax.whiten(xs[row])
    plt.plot(range(0, len(xs))[row], ss)
    for index1, row1 in enumerate(normal_ppa_result[index]):
        pos_coll = pos_coll_main + index1 * len(xs[row]) / nbins
        pos_coll_nex = pos_coll_main + (index1 + 1) * len(xs[row]) / nbins
        if pos_coll == pos_coll_main:
            plt.axvline(x=pos_coll, linewidth=2, color='#000000', linestyle='dashdot')
            # else:
            # plt.axvline(x = pos_coll, linewidth=2, color='#d62728',linestyle='dashdot')
        plt.plot((pos_coll + pos_coll_nex) / 2, row1, 'bo', color='green', markersize=10)

letter_array1 = sax.cutpoints[len(alphabet)]
letter_array = filter(lambda x: np.isinf(x) != True, letter_array1)
# print letter_array
j = 0
z = 0
for i in letter_array:
    plt.axhline(y=i, linewidth=2, color='#1f77b4', linestyle='dashdot')
    plt.text(0.2, i - 0.3, alphabet[j], fontsize=15, color='#d62728')
    j = j + 1
    z = i
plt.text(0.2, z + 0.3, alphabet[j], fontsize=15, color='#d62728')
# plt.grid(True)

plt.show

# In[346]:

from numpy import array


def mean_distance_symbol(a, array_of_a):
    return np.mean(map(lambda x: sax.symbol_distance(x, a), array_of_a))


########################### Optimized Case #######################################
mean_SAX_result_array = map(lambda x: array(([np.min(sax.symbol_to_vector(x)), np.max(sax.symbol_to_vector(x)),
                                              np.mean(sax.symbol_to_vector(x)), np.std(sax.symbol_to_vector(x))])),
                            SAX_result)
print "Clustering vector for optimized Case - without normalize \n", mean_SAX_result_array

mean_SAX_result_array = map(lambda x: array(sax.whiten(
    [np.min(sax.symbol_to_vector(x)), np.max(sax.symbol_to_vector(x)), np.mean(sax.symbol_to_vector(x)),
     np.std(sax.symbol_to_vector(x))])), SAX_result)
print "Clustering vector for optimized Case \n", mean_SAX_result_array

########################### Normal Case #########################################

mean_normal_SAX_result_array = map(lambda x: array(sax.whiten(
    [np.min(sax.symbol_to_vector(x)), np.max(sax.symbol_to_vector(x)), np.mean(sax.symbol_to_vector(x)),
     np.std(sax.symbol_to_vector(x))])), normal_sax_result)
print "Clustering vector for Normal Case \n", mean_normal_SAX_result_array

# sax.symbol_distance(SAX_result[0],SAX_result[1])
# mean_distance_symbol(x, SAX_result),


# In[347]:

from sklearn.cluster import KMeans

# In[348]:

input_data = np.array(mean_SAX_result_array)
normal_input_data = np.array(mean_normal_SAX_result_array)
# print input_data_1
kmeans = KMeans(n_clusters=3, random_state=0).fit(input_data)
normal_kmeans = KMeans(n_clusters=3, random_state=0).fit(normal_input_data)
print "Optimized Clustering result:", kmeans
print "Normal Clustering result:", normal_kmeans

# In[349]:

labels = kmeans.labels_
normal_labels = normal_kmeans.labels_
print "Label in optimized case: \n", labels
print "Label in normal case: \n", normal_labels

# In[350]:

center_point = kmeans.cluster_centers_
normal_center_point = normal_kmeans.cluster_centers_

# In[351]:

print "Mean distance in optimized case:", kmeans.inertia_ / len(labels)
print "Mean of distance in normal case:", normal_kmeans.inertia_ / len(normal_labels)

# In[352]:

# plot the clusters in color
for i, index in enumerate(SAX_result):
    print "Label: %d - Value: %s" % (labels[int(i)], index)
fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()
iris = np.array(mean_SAX_result_array)
ax.scatter(input_data[:, [0]], input_data[:, [1]], input_data[:, [2]], c=labels)

# moon
# np.random.seed(0)
# X, y = datasets.make_moons(2000, noise=0.2)

# blob
# np.random.seed(0)
# X, y = datasets.make_blobs(n_samples=2000, centers=3, n_features=20, random_state=0)

# centers = kmeans(X, k=3)
# labels = [find_closest_centroid(p, centers) for p in X]

# fig = plt.figure(1, figsize=(8, 8))
# plt.clf()
# plt.scatter(X[:,0], X[:,1], s=40, c=labels, cmap=plt.cm.Spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Mean Vector')
ax.set_ylabel('Min Vector')
ax.set_zlabel('Max Vector')

plt.show()

################################################## NORMAL CASE ########################################################
# plot the clusters in color
for i, index in enumerate(normal_sax_result):
    print "Label: %d - Value: %s" % (normal_labels[int(i)], index)
fig = plt.figure(2, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()
iris = np.array(mean_normal_SAX_result_array)
ax.scatter(normal_input_data[:, [0]], normal_input_data[:, [1]], normal_input_data[:, [2]], c=normal_labels)

# moon
# np.random.seed(0)
# X, y = datasets.make_moons(2000, noise=0.2)

# blob
# np.random.seed(0)
# X, y = datasets.make_blobs(n_samples=2000, centers=3, n_features=20, random_state=0)

# centers = kmeans(X, k=3)
# labels = [find_closest_centroid(p, centers) for p in X]

# fig = plt.figure(1, figsize=(8, 8))
# plt.clf()
# plt.scatter(X[:,0], X[:,1], s=40, c=labels, cmap=plt.cm.Spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Mean Vector')
ax.set_ylabel('Min Vector')
ax.set_zlabel('Max Vector')

plt.show()


'''