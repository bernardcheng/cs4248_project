from collections import defaultdict

import csv
import re
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .languages import CORE_LANGUAGES
from .relations import SYMMETRIC_RELATIONS
from .uri import get_uri_language, uri_prefix, uri_prefixes
from ordered_set import OrderedSet

DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
DIGIT_RE = re.compile(r'[0-9]')
CONCEPT_RE = re.compile(r'/c/[a-z]{2,3}/.+')


def replace_numbers(s):
    """
    Replace digits with # in any term where a sequence of two digits appears.

    This operation is applied to text that passes through word2vec, so we
    should match it.
    """
    if DOUBLE_DIGIT_RE.search(s):
        return DIGIT_RE.sub('#', s)
    else:
        return s


class SparseMatrixBuilder:
    """
    SparseMatrixBuilder is a utility class that helps build a matrix of
    unknown shape.
    """

    def __init__(self):
        self.row_index = []
        self.col_index = []
        self.values = []

    def __setitem__(self, key, val):
        row, col = key
        self.add(row, col, val)

    def add(self, row, col, val):
        self.row_index.append(row)
        self.col_index.append(col)
        self.values.append(val)

    def tocsr(self, shape, dtype=float):
        return sparse.coo_matrix(
            (self.values, (self.row_index, self.col_index)), shape=shape, dtype=dtype
        ).tocsr()


def build_from_conceptnet_table(filename, orig_index=(), self_loops=True):
    """
    Read a file of tab-separated association data from ConceptNet, such as
    `data/assoc/reduced.csv`. Return a SciPy sparse matrix of the associations,
    and a pandas Index of labels.

    If you specify `orig_index`, then the index of labels will be pre-populated
    with existing labels, and any new labels will get index numbers that are
    higher than the index numbers the existing labels use. This is important
    for producing a sparse matrix that can be used for retrofitting onto an
    existing dense labeled matrix (see retrofit.py).
    """
    mat = SparseMatrixBuilder()

    labels = OrderedSet(orig_index)

    totals = defaultdict(float)
    with open(str(filename), encoding='utf-8') as infile:        
        reader = csv.DictReader(infile)  # automatically skips header
        for row in reader:
            end_concept = row['end_id']
            start_concept = row['start_id']
            relation = row['rel_id']
            value_str = row['weight']
            dataset = row['dataset']
            
            index1 = labels.add(replace_numbers(end_concept))
            index2 = labels.add(replace_numbers(start_concept))
            try:
                value = float(value_str)
            except:
                print(value_str)
                value = 0.0

            mat[index1, index2] = value
            mat[index2, index1] = value
            totals[index1] += value
            totals[index2] += value

    # Link nodes to their more general versions
    for label in labels:
        prefixes = list(uri_prefixes(label, 3))
        if len(prefixes) >= 2:
            parent_uri = prefixes[-2]
            if parent_uri in labels:
                index1 = labels.index(label)
                index2 = labels.index(parent_uri)
                mat[index1, index2] = 1
                mat[index2, index1] = 1
                totals[index1] += 1
                totals[index2] += 1

    # add self-loops on the diagonal with equal weight to the rest of the row
    if self_loops:
        for key, value in totals.items():
            mat[key, key] = value

    shape = (len(labels), len(labels))
    index = pd.Index(labels)
    return normalize(mat.tocsr(shape), norm='l1', axis=1), index


def build_features_from_conceptnet_table(filename):
    mat = SparseMatrixBuilder()

    concept_labels = OrderedSet()
    feature_labels = OrderedSet()

    with open(str(filename), encoding='utf-8') as infile:
        next(infile) #start reading from line 2 (skip header row)
        for line in infile:
            end_concept, _, start_concept, _, relation, _, value_str, dataset = line.strip().split(',')

            end_concept = replace_numbers(end_concept)
            start_concept = replace_numbers(start_concept)
            value = float(value_str)
            if relation in SYMMETRIC_RELATIONS:
                feature_pairs = []
                if get_uri_language(end_concept) in CORE_LANGUAGES:
                    feature_pairs.append(
                        ('{} {} ~'.format(uri_prefix(end_concept), relation), start_concept)
                    )
                if get_uri_language(start_concept) in CORE_LANGUAGES:
                    feature_pairs.append(
                        ('{} {} ~'.format(uri_prefix(start_concept), relation), end_concept)
                    )
            else:
                if get_uri_language(end_concept) in CORE_LANGUAGES:
                    feature_pairs.append(
                        ('{} {} -'.format(uri_prefix(end_concept), relation), start_concept)
                    )
                if get_uri_language(start_concept) in CORE_LANGUAGES:
                    feature_pairs.append(
                        ('- {} {}'.format(uri_prefix(start_concept), relation), end_concept)
                    )

            feature_counts = defaultdict(int)
            for feature, concept in feature_pairs:
                feature_counts[feature] += 1

            for feature, concept in feature_pairs:
                prefixes = list(uri_prefixes(concept, 3))
                if feature_counts[feature] > 1:
                    for prefix in prefixes:
                        concept_index = concept_labels.add(prefix)
                        feature_index = feature_labels.add(feature)
                        mat[concept_index, feature_index] = value

    # Link nodes to their more general versions
    for concept in concept_labels:
        prefixes = list(uri_prefixes(concept, 3))
        for prefix in prefixes:
            auto_features = [
                '{} {} ~'.format(prefix, 'SimilarTo'),
                '{} {} ~'.format(prefix, 'RelatedTo'),
                '{} {} -'.format(prefix, 'FormOf'),
                '- {} {}'.format(prefix, 'FormOf'),
            ]
            for feature in auto_features:
                concept_index = concept_labels.add(prefix)
                feature_index = feature_labels.add(feature)
                mat[concept_index, feature_index] = value

    shape = (len(concept_labels), len(feature_labels))
    c_index = pd.Index(concept_labels)
    f_index = pd.Index(feature_labels)
    return mat.tocsr(shape), c_index, f_index