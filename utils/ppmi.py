import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg

from utils.sparse_matrix_builder import build_from_conceptnet_table

def build_ppmi(conceptnet_filename, ndim=300):
    sparse_csr, index = build_from_conceptnet_table(conceptnet_filename)
    ppmi = counts_to_ppmi(sparse_csr)
    u, s, vT = linalg.svds(ppmi, ndim)
    v = vT.T
    values = (u + v) * (s ** 0.5)

    return pd.DataFrame(values, index=index)


def counts_to_ppmi(counts_csr, smoothing=0.75):
    """
    Converts a sparse matrix of co-occurrences into a sparse matrix of positive
    pointwise mutual information. Context distributional smoothing is applied
    to the resulting matrix.
    """
    # word_counts adds up the total amount of association for each term.
    word_counts = np.asarray(counts_csr.sum(axis=1)).flatten()

    # smooth_context_freqs represents the relative frequency of occurrence
    # of each term as a context (a column of the table).
    smooth_context_freqs = np.asarray(counts_csr.sum(axis=0)).flatten() ** smoothing
    smooth_context_freqs /= smooth_context_freqs.sum()

    # Divide each row of counts_csr by the word counts. We accomplish this by
    # multiplying on the left by the sparse diagonal matrix of 1 / word_counts.
    ppmi = sparse.diags(1 / word_counts).dot(counts_csr)

    # Then, similarly divide the columns by smooth_context_freqs, by the same
    # method except that we multiply on the right.
    ppmi = ppmi.dot(sparse.diags(1 / smooth_context_freqs))

    # Take the log of the resulting entries to give pointwise mutual
    # information. Discard those whose PMI is less than 0, to give positive
    # pointwise mutual information (PPMI).
    ppmi.data = np.maximum(np.log(ppmi.data), 0)
    ppmi.eliminate_zeros()
    return ppmi