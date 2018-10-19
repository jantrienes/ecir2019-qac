import argparse
import csv
import glob
import itertools
import logging
import logging.config
import os
from collections import Counter, defaultdict
from os.path import basename, dirname, exists, join, splitext

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import spatial, stats
from tqdm import tqdm

from qac.experiments import preprocessing, rake, readability

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def run_path(run_id):
    return join(dirname(__file__), '../../models/simq', run_id)


def load_run(run_file):
    trec_retrieved = defaultdict(list)  # query_id -> [docid1, docid2, ...]
    trec_scores = defaultdict(list)  # query_id -> [d1_score, d2_score, ...]
    with open(run_file) as file:
        for line in file:
            (qid, _, doc_id, _, score, _) = line.strip().split()
            qid, doc_id = int(qid), int(doc_id)
            if qid == doc_id:
                # Exclude the original post from the set of similar questions.
                # Happens for posts of the training data as they have been indexed.
                continue
            trec_retrieved[qid].append(doc_id)
            trec_scores[qid].append(float(score))
    return trec_retrieved, trec_scores


# pylint: disable=unused-argument
class FeatureGenerator():

    def __init__(self, df_posts, df_clarqs, clear_lbl=0, unclear_lbl=1):
        """
        :df_posts: Pandas dataframe with columns [id, title, body, tags, label]. Post ID shall be
        set as index column.
        """
        self._data = df_posts
        self._clarqs = df_clarqs
        self._clear_lbl = clear_lbl
        self._unclear_lbl = unclear_lbl

        self._rake = rake.Rake()

    def feature(self, p_id, q_ids, q_scores):
        """
        Generates a feature for a post p and given similar questions q.

        :p_id: ID of post for which this feature is generated.
        :q_ids: ID's of similar posts.
        :q_scores: Retrieval scores of questions similar to p.
        """
        pass

    def feat_post_length(self, p_id, q_ids, q_scores):
        post = self._tokenized_post(p_id)
        return len(post)

    def _body_matches_regex(self, p_id, regex):
        post = self._data.body.loc[p_id]
        match = regex.search(post)
        return True if match else False

    def feat_post_contains_preformatted(self, p_id, q_ids, q_scores):
        return self._body_matches_regex(p_id, preprocessing.CODE_REGEX)

    def feat_post_contains_blockquote(self, p_id, q_ids, q_scores):
        return self._body_matches_regex(p_id, preprocessing.BLOCKQUOTE_REGEX)

    def feat_post_contains_questionmark(self, p_id, q_ids, q_scores):
        post = self._data.body.loc[p_id]
        return '?' in post

    def feat_post_readability(self, p_id, q_ids, q_scores):
        post = self._data.body.loc[p_id]
        analyzer = readability.ReadabilityAnalyzer(post)
        return analyzer.coleman_liau_index

    @staticmethod
    def feat_sim_sum(p_id, q_ids, q_scores):
        return sum(q_scores)

    @staticmethod
    def feat_sim_max(p_id, q_ids, q_scores):
        return np.max(q_scores)

    @staticmethod
    def feat_sim_avg(p_id, q_ids, q_scores):
        return np.mean(q_scores)

    @staticmethod
    def feat_num_similar(p_id, q_ids, q_scores):
        return len(q_ids)

    def _similar_label_count(self, q_ids):
        similar_labels = self._data.label.loc[q_ids].tolist()
        clear = similar_labels.count(self._clear_lbl)
        unclear = similar_labels.count(self._unclear_lbl)
        return clear, unclear

    def feat_num_clear(self, p_id, q_ids, q_scores):
        clear, _ = self._similar_label_count(q_ids)
        return clear

    def feat_num_unclear(self, p_id, q_ids, q_scores):
        _, unclear = self._similar_label_count(q_ids)
        return unclear

    def feat_majority(self, p_id, q_ids, q_scores):
        similar_labels = self._data.label.loc[q_ids].values
        return stats.mode(similar_labels).mode[0]

    def feat_ratio(self, p_id, q_ids, q_scores):
        clear, unclear = self._similar_label_count(q_ids)

        if unclear == 0:
            return clear

        return clear / unclear

    def feat_fraction(self, p_id, q_ids, q_scores):
        similar_labels = self._data.label.loc[q_ids].tolist()
        clear = similar_labels.count(self._clear_lbl)
        total = len(q_ids)
        return clear / total

    def feat_global_cos(self, p_id, q_ids, q_scores):
        post_tokens = self._tokenized_post(p_id)
        cq = self._clarqs.loc[q_ids]['clarification_question'].values
        cq_subjects = [self._subjects(clarq) for clarq in cq]

        p_vec, cq_vec = self.vectorize_subjects(post_tokens, cq_subjects)
        return self.score_cosine(p_vec, cq_vec)

    def feat_individual_cos(self, p_id, q_ids, q_scores):
        return self.feat_individual_cos_weighted(p_id, q_ids, [1] * len(q_ids))

    def feat_individual_cos_weighted(self, p_id, q_ids, q_scores):
        post_tokens = self._tokenized_post(p_id)
        cq = self._clarqs.loc[q_ids]['clarification_question'].values

        score = 0
        for clarq, weight in zip(cq, q_scores):
            cq_subjects = self._subjects(clarq)
            p_vec, cq_vec = self.vectorize_subjects(post_tokens, [cq_subjects])
            score += self.score_cosine(p_vec, cq_vec) * weight
        return score

    def _subjects(self, clarq):
        chunks = self._rake.run(clarq)
        tokens = [token for chunk, _ in chunks for token in chunk.split()]
        return tokens

    def _tokenized_post(self, p_id):
        p = self._data.loc[p_id]
        combined = '{} {} {}'.format(p.title, p.body, p.tags)
        post_tokens = preprocessing.get_tokens(combined)
        return post_tokens

    @staticmethod
    def vectorize_subjects(p, cq):
        """
        Vectorizes a post and clarification questions. A vocabulary is constructed based on the
        tokens in cq.

        :param p: Tokenized post
        :param cq: List of tokenized clarification subjects
        :type p: list(str)
        :type cq: list(list(str))
        """
        p_counts = Counter(p)
        subjects = Counter(itertools.chain.from_iterable(cq))

        subject_keys = sorted(subjects.keys())
        p_vec = [p_counts.get(key, 0) for key in subject_keys]
        cq_vec = [subjects.get(key) for key in subject_keys]

        return p_vec, cq_vec

    @staticmethod
    def score_cosine(a, b):
        """
        Adapted cosine similarity: if either input vector is all zero, the similarity is defined to
        be zero.
        """
        sim = 1 - spatial.distance.cosine(a, b)
        if np.isnan(sim):
            sim = 0
        return sim
# pylint: enable=unused-argument


class FeatureHandler():

    def __init__(self, community, run, out_path=None):
        self.community = community
        self.run = run
        if out_path is None:
            out_path = '../../models/simq-features/{}/{}/'.format(community, run)
            out_path = join(dirname(__file__), out_path)
            os.makedirs(out_path, exist_ok=True)
        self.out_path = out_path

    def exists(self, feature_name):
        out_file = '{}.csv'.format(feature_name)
        return exists(join(self.out_path, out_file))

    def write(self, feature_name, ids, values):
        out_file = '{}.csv'.format(feature_name)

        with open(join(self.out_path, out_file), 'w') as file:
            writer = csv.writer(file)
            for post_id, value in zip(ids, values):
                writer.writerow([post_id, value])

    def read(self, feature_name):
        filename = '{}.csv'.format(feature_name)
        data = pd.read_csv(join(self.out_path, filename), header=None, names=['id', feature_name])
        data.set_index('id', inplace=True)
        return data

    def read_all(self):
        all_features = glob.glob(join(self.out_path, '*.csv'))
        feats = (pd.read_csv(f, header=None, index_col=0, names=['id', splitext(basename(f))[0]])
                 for f in all_features)
        return pd.concat(feats, axis=1)


def _features_post(feature_generator):
    def feat_post_length(p_id, q_ids, q_scores):
        return feature_generator.feat_post_length(p_id, q_ids, q_scores)

    def feat_post_contains_preformatted(p_id, q_ids, q_scores):
        return feature_generator.feat_post_contains_preformatted(p_id, q_ids, q_scores)

    def feat_post_contains_blockquote(p_id, q_ids, q_scores):
        return feature_generator.feat_post_contains_blockquote(p_id, q_ids, q_scores)

    def feat_post_contains_questionmark(p_id, q_ids, q_scores):
        return feature_generator.feat_post_contains_questionmark(p_id, q_ids, q_scores)

    def feat_post_readability(p_id, q_ids, q_scores):
        return feature_generator.feat_post_readability(p_id, q_ids, q_scores)

    return [
        feat_post_length,
        feat_post_contains_preformatted,
        feat_post_contains_blockquote,
        feat_post_contains_questionmark,
        feat_post_readability
    ]


def _features_all(feature_generator):
    def feat_all_majority_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_majority(p_id, q_ids[:10], q_scores[:10])

    def feat_all_majority_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_majority(p_id, q_ids[:20], q_scores[:20])

    def feat_all_majority_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_majority(p_id, q_ids[:50], q_scores[:50])

    def feat_all_fraction_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_fraction(p_id, q_ids[:10], q_scores[:10])

    def feat_all_fraction_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_fraction(p_id, q_ids[:20], q_scores[:20])

    def feat_all_fraction_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_fraction(p_id, q_ids[:50], q_scores[:50])

    def feat_all_ratio_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_ratio(p_id, q_ids[:10], q_scores[:10])

    def feat_all_ratio_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_ratio(p_id, q_ids[:20], q_scores[:20])

    def feat_all_ratio_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_ratio(p_id, q_ids[:50], q_scores[:50])

    def feat_all_num_similar_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_num_similar(p_id, q_ids[:10], q_scores[:10])

    def feat_all_num_similar_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_num_similar(p_id, q_ids[:20], q_scores[:20])

    def feat_all_num_similar_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_num_similar(p_id, q_ids[:50], q_scores[:50])

    def feat_all_num_clear_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_num_clear(p_id, q_ids[:10], q_scores[:10])

    def feat_all_num_clear_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_num_clear(p_id, q_ids[:20], q_scores[:20])

    def feat_all_num_clear_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_num_clear(p_id, q_ids[:50], q_scores[:50])

    def feat_all_num_unclear_k10(p_id, q_ids, q_scores):
        return feature_generator.feat_num_unclear(p_id, q_ids[:10], q_scores[:10])

    def feat_all_num_unclear_k20(p_id, q_ids, q_scores):
        return feature_generator.feat_num_unclear(p_id, q_ids[:20], q_scores[:20])

    def feat_all_num_unclear_k50(p_id, q_ids, q_scores):
        return feature_generator.feat_num_unclear(p_id, q_ids[:50], q_scores[:50])

    def feat_all_sim_sum(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_sum(p_id, q_ids, q_scores)

    def feat_all_sim_max(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_max(p_id, q_ids, q_scores)

    def feat_all_sim_avg(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_avg(p_id, q_ids, q_scores)

    return [
        feat_all_majority_k10,
        feat_all_majority_k20,
        feat_all_majority_k50,
        feat_all_fraction_k10,
        feat_all_fraction_k20,
        feat_all_fraction_k50,
        feat_all_ratio_k10,
        feat_all_ratio_k20,
        feat_all_ratio_k50,
        feat_all_num_similar_k10,
        feat_all_num_similar_k20,
        feat_all_num_similar_k50,
        feat_all_num_clear_k10,
        feat_all_num_clear_k20,
        feat_all_num_clear_k50,
        feat_all_num_unclear_k10,
        feat_all_num_unclear_k20,
        feat_all_num_unclear_k50,
        feat_all_sim_sum,
        feat_all_sim_max,
        feat_all_sim_avg
    ]


def _features_unclear(feature_generator):
    def feat_unclear_global_cos(p_id, q_ids, q_scores):
        return feature_generator.feat_global_cos(p_id, q_ids[:10], q_scores[:10])

    def feat_unclear_individual_cos(p_id, q_ids, q_scores):
        return feature_generator.feat_individual_cos(p_id, q_ids[:10], q_scores[:10])

    def feat_unclear_individual_cos_weighted(p_id, q_ids, q_scores):
        return feature_generator.feat_individual_cos_weighted(p_id, q_ids[:10], q_scores[:10])

    def feat_unclear_sim_sum(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_sum(p_id, q_ids, q_scores)

    def feat_unclear_sim_max(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_max(p_id, q_ids, q_scores)

    def feat_unclear_sim_avg(p_id, q_ids, q_scores):
        return feature_generator.feat_sim_avg(p_id, q_ids, q_scores)

    return [
        feat_unclear_global_cos,
        feat_unclear_individual_cos,
        feat_unclear_individual_cos_weighted,
        feat_unclear_sim_sum,
        feat_unclear_sim_max,
        feat_unclear_sim_avg
    ]


def _compute_feature(feature, feature_writer, p_ids, similar, scores, K=50):
    """
    :param K: Restrict set of similar questions to the top K similar.
    :type K: int
    """
    values = []
    for p_id in p_ids:
        try:
            q_ids = similar[p_id][:K]
            q_scores = scores[p_id][:K]
            values.append(feature(p_id, q_ids, q_scores))
        except Exception as e:
            logger.exception(e)
            logger.warning('Exception during computation of feature "%s" for post "%s"',
                           feature.__name__, p_id)

    feature_writer.write(feature.__name__, p_ids, values)


def get_question(clarq_comment):
    """Extract the question part of a clarification question comment. Only alphabetic tokens are
    retained.

    Example:
        "I dont understand what you are doing. Can you add code?" => "can you add code"
    """
    tokens = preprocessing.get_tokens(clarq_comment)
    end = tokens.index('?') + 1
    context = tokens[:end][::-1]

    start = 0
    if '.' in context:
        # remove context
        start = end - context.index('.')

    clarq = tokens[start:end]
    clarq = [token for token in clarq if token.isalpha()]
    return ' '.join(clarq)


def filter_features(feature_funcs, feature_writer):
    filtered = []
    for feature in feature_funcs:
        if feature_writer.exists(feature.__name__):
            logger.info('"%s" already exists. Skip computation...', feature.__name__)
        else:
            filtered.append(feature)
    return filtered


def _compute_sequential(features, feature_writer, p_ids, similar, scores):
    for feature in features:
        logger.info('Compute feature: %s', feature.__name__)
        _compute_feature(feature, feature_writer, tqdm(p_ids), similar, scores)


def _compute_parallel(features, feature_writer, p_ids, similar, scores, n_jobs):
    # pylint: disable=E1123
    with Parallel(n_jobs=n_jobs, prefer="processes", verbose=1) as parallel:
        parallel(delayed(_compute_feature)(feature, feature_writer, p_ids,
                                           similar, scores)
                 for feature in features)


def main(args):
    logger.info('Load data and clarification questions...')
    df_posts = preprocessing.load_community_raw(args.community)
    df_posts.set_index('id', inplace=True)

    df_clarqs = preprocessing.load_clarqs(args.community)
    df_clarqs['clarification_question'] = df_clarqs['clarification_question'].apply(get_question)
    df_clarqs.set_index('id', inplace=True)

    feature_generator = FeatureGenerator(df_posts, df_clarqs)
    feature_writer = FeatureHandler(args.community, args.run_id)

    logger.info('Start feature computation...')
    features_all = filter_features(_features_all(feature_generator), feature_writer)
    features_unclear = filter_features(_features_unclear(feature_generator), feature_writer)
    features_post = filter_features(_features_post(feature_generator), feature_writer)
    features_unclear += features_post  # can be computed on any set of similar questions

    logger.info('Load retrieval results for "all" run...')
    similar_all, scores_all = load_run(
        run_path('stackexchange_{}_{}.run'.format(args.community, args.run_id)))
    logger.info('Compute "%d" features', len(features_all))
    if args.community == 'stackoverflow':
        _compute_sequential(features_all, feature_writer, df_posts.index.values, similar_all,
                            scores_all)
    else:
        _compute_parallel(features_all, feature_writer, df_posts.index.values, similar_all,
                          scores_all, args.n_jobs)

    del similar_all
    del scores_all
    logger.info('Load retrieval results for "unclear" run...')
    similar_unclear, scores_unclear = load_run(
        run_path('stackexchange_{}_unclear_{}.run'.format(args.community, args.run_id)))
    logger.info('Compute "%d" features', len(features_unclear))
    if args.community == 'stackoverflow':
        _compute_sequential(features_unclear, feature_writer, df_posts.index.values,
                            similar_unclear, scores_unclear)
    else:
        _compute_parallel(features_unclear, feature_writer, df_posts.index.values,
                          similar_unclear, scores_unclear, args.n_jobs)
    logger.info('Finish feature computation.')


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("run_id", help="Identifier of similar question retrieval run", type=str)
    parser.add_argument("--n_jobs", help="Number of jobs in parallel feature computation", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    import nltk
    nltk.download('punkt')  # for readability analysis

    logging.config.fileConfig(join(dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
