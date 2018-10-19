import pandas as pd
import pytest

from qac.simq.simq_features import FeatureGenerator

DATA = pd.DataFrame([
    [0, 'title', 'body', 'tags', 1],
    [1, 'title', 'body', 'tags', 1],
    [2, 'title', 'body', 'tags', 1],
    [3, 'title', 'body', 'tags', 1],
    [4, 'title', 'body', 'tags', 1],
    [5, 'title', 'body', 'tags', 0],
    [6, 'title', 'body', 'tags', 0],
    [7, 'title', 'body', 'tags', 0],
    [8, 'title', 'body', 'tags', 0],
    [9, 'title', 'body', 'tags', 0]
], columns=['id', 'title', 'body', 'tags', 'label'])
DATA.set_index('id', inplace=True)

CLARQS = pd.DataFrame([
    [0, "is this a question?"]
], columns=['id', 'clarification_question'])
CLARQS.set_index('id', inplace=True)

DATA_SCORING = pd.DataFrame([
    [0, 't1', 't0 t1 t2 t3', 'tag1 tag2', 0],
], columns=['id', 'title', 'body', 'tags', 'label'])
DATA_SCORING.set_index('id', inplace=True)

CLARQS_SCORING = pd.DataFrame([
    [1, 't4 t1'],   # cos([0, 2], [1, 1]) = 0.707
    [2, 't1'],      # cos([2], [1]) = 1
    [3, 't2'],      # cos([1], [1]) = 1
    [4, 't4'],      # cos([0], [1]) := 0
], columns=['id', 'clarification_question'])
CLARQS_SCORING.set_index('id', inplace=True)


def test_feat_majority():
    generator = FeatureGenerator(DATA, CLARQS)

    majority = generator.feat_majority(0, [0, 1, 4, 5, 6], [100, 3, 2, 1, 0.5])
    assert majority == 1

    majority = generator.feat_majority(0, [0, 1, 5, 6, 7], [100, 3, 2, 1, 0.5])
    assert majority == 0

    majority = generator.feat_majority(0, [0, 1, 4, 5, 6, 7], [100, 3, 2, 1, 0.5])
    assert majority == 0

    majority = generator.feat_majority(0, [5, 6, 7, 0, 1, 4], [100, 3, 2, 1, 0.5])
    assert majority == 0


def test_feat_ratio():
    generator = FeatureGenerator(DATA, CLARQS)

    ratio = generator.feat_ratio(0, [0, 1, 5, 6], [100, 3, 2, 1, 0.5])
    assert ratio == 1

    ratio = generator.feat_ratio(0, [5, 6, 7, 0, 1], [100, 3, 2, 1, 0.5])
    assert ratio == 1.5

    ratio = generator.feat_ratio(0, [5, 6, 0, 1, 2, 3], [100, 3, 2, 1, 0.5])
    assert ratio == 0.5

    ratio = generator.feat_ratio(0, [0], [100, 3, 2, 1, 0.5])
    assert ratio == 0

    # return number of clear if there are no unclear similar questions
    ratio = generator.feat_ratio(0, [6, 7, 8], [100, 3, 2, 1, 0.5])
    assert ratio == 3


def test_feat_fraction():
    generator = FeatureGenerator(DATA, CLARQS)

    proportion = generator.feat_fraction(0, [7, 0, 1, 2, 3], [100, 3, 2, 1, 0.5])
    assert proportion == 0.2

    proportion = generator.feat_fraction(0, [5], [100, 3, 2, 1, 0.5])
    assert proportion == 1

    proportion = generator.feat_fraction(0, [0], [100, 3, 2, 1, 0.5])
    assert proportion == 0


def test_vectorize_subjects():
    # tokenized post and clarification questions
    p = ['t1', 't1', 't2']
    cq1 = ['t4', 't1']
    cq2 = ['t1']
    cq3 = ['t2']
    cq4 = ['t4']
    cq = [cq1, cq2, cq3, cq4]

    p_vec, cq_vec = FeatureGenerator.vectorize_subjects(p, cq)

    assert p_vec == [2, 1, 0]
    assert cq_vec == [2, 1, 2]


def test_score_cosine():
    p_vec = [2, 1, 0]
    cq_vec = [2, 1, 2]

    assert FeatureGenerator.score_cosine(p_vec, cq_vec) == pytest.approx(0.745, rel=1e-3)


def test_feat_global_cos():
    generator = FeatureGenerator(DATA_SCORING, CLARQS_SCORING)
    p_id = 0
    q_ids = [1, 2, 3, 4]
    q_scores = [1, 1, 1, 1]
    assert generator.feat_global_cos(p_id, q_ids, q_scores) == pytest.approx(0.745, rel=1e-3)


def test_feat_individual_cos():
    generator = FeatureGenerator(DATA_SCORING, CLARQS_SCORING)
    p_id = 0
    q_ids = [1, 2, 3, 4]
    q_scores = [1, 1, 1, 1]
    assert generator.feat_individual_cos(p_id, q_ids, q_scores) == pytest.approx(2.707, rel=1e-3)


def test_feat_individual_weighted_cos():
    generator = FeatureGenerator(DATA_SCORING, CLARQS_SCORING)
    p_id = 0
    q_ids = [1, 2, 3, 4]
    q_scores = [10, 2, 1, 1]
    score = generator.feat_individual_cos_weighted(p_id, q_ids, q_scores)
    assert score == pytest.approx(10.07, rel=1e-2)


def test_feat_post_length():
    data = pd.DataFrame([
        [0, 'title', 'body1 body2 body3', 'tags', 1],
    ], columns=['id', 'title', 'body', 'tags', 'label'])
    data.set_index('id', inplace=True)

    generator = FeatureGenerator(data, df_clarqs=None)
    length = generator.feat_post_length(p_id=0, q_ids=[], q_scores=[])
    assert length == 5


def test_feat_sim_sum():
    sim_sum = FeatureGenerator.feat_sim_sum(p_id=None, q_ids=[], q_scores=[5, 4, 3, 2, 1])
    assert sim_sum == 15


def test_feat_sim_max():
    sim_max = FeatureGenerator.feat_sim_max(p_id=None, q_ids=[], q_scores=[5, 4, 3, 2, 1])
    assert sim_max == 5


def test_feat_sim_avg():
    sim_avg = FeatureGenerator.feat_sim_avg(p_id=None, q_ids=[], q_scores=[5, 4, 3, 2, 1])
    assert sim_avg == 3


def test_feat_num_similar():
    num_similar = FeatureGenerator.feat_num_similar(
        p_id=None, q_ids=[1, 2, 3, 4, 5], q_scores=[5, 4, 3, 2, 1])
    assert num_similar == 5


def _two_post_df(post1, post2):
    data = pd.DataFrame([
        [0, 'title', post1, 'tags', 1],
        [1, 'title', post2, 'tags', 1],
    ], columns=['id', 'title', 'body', 'tags', 'label'])
    data.set_index('id', inplace=True)

    return data


def test_feat_post_contains_preformatted():
    post1 = "this is first line\n    code\n    more code\n    more code\nno code"
    post2 = "this is first line\nand post has no code in it"
    data = _two_post_df(post1, post2)

    generator = FeatureGenerator(data, df_clarqs=None)
    assert generator.feat_post_contains_preformatted(p_id=0, q_ids=[], q_scores=[]) is True
    assert generator.feat_post_contains_preformatted(p_id=1, q_ids=[], q_scores=[]) is False


def test_feat_post_contains_questionmark():
    post1 = "this is first line"
    post2 = "this is my question?"
    data = _two_post_df(post1, post2)

    generator = FeatureGenerator(data, df_clarqs=None)
    assert generator.feat_post_contains_questionmark(p_id=0, q_ids=[], q_scores=[]) is False
    assert generator.feat_post_contains_questionmark(p_id=1, q_ids=[], q_scores=[]) is True


def test_feat_post_contains_blockquote():
    post1 = "this is first >line"
    post2 = "this is my question?\n> some >test"
    data = _two_post_df(post1, post2)

    generator = FeatureGenerator(data, df_clarqs=None)
    assert generator.feat_post_contains_blockquote(p_id=0, q_ids=[], q_scores=[]) is False
    assert generator.feat_post_contains_blockquote(p_id=1, q_ids=[], q_scores=[]) is True


def test_compute_paper_example():
    title = 'Simplest XML editor'
    body = "I need the simplest editor with utf8 support for editing xml files; It's for a non programmer (so no atom or the like), to edit existing files. Any suggestion?"
    tags = '<xml><utf8><editors>'

    data = pd.DataFrame([
        [0, title, body, tags, None],
        [1, 'title', 'body', 'tags', 1],
        [2, 'title', 'body', 'tags', 0],
        [3, 'title', 'body', 'tags', 1],
    ], columns=['id', 'title', 'body', 'tags', 'label'])
    data.set_index('id', inplace=True)

    clarqs = pd.DataFrame([
        [1, "What operating system?"],
        [3, "Have you tried atom?"]
    ], columns=['id', 'clarification_question'])
    clarqs.set_index('id', inplace=True)

    generator = FeatureGenerator(data, clarqs)

    p_id = 0
    q_ids = [1, 2, 3]
    q_scores = [5, 2, 1]

    def blank(p_id, q_ids, q_scores): return ''
    feature_table = [
        ('(i) Features on q', blank),
        ('==================', blank),
        ('Len(q)', generator.feat_post_length),
        ('ContainsPre(q)', generator.feat_post_contains_preformatted),
        ('ContainsQuote(q)', generator.feat_post_contains_blockquote),
        ('ContainsQuest(q)', generator.feat_post_contains_questionmark),
        ('Readability(q)', generator.feat_post_readability),
        ('(ii) Features on Q\'', blank),
        ('==================', blank),
        ('SimSum(q,Q\')', generator.feat_sim_sum),
        ('SimMax(q,Q\')', generator.feat_sim_max),
        ('SimAvg(q,Q\')', generator.feat_sim_avg),
        ('LenSim(Q\')', generator.feat_num_similar),
        ('SimUnclear(Q\')', generator.feat_num_unclear),
        ('SimClear(Q\')', generator.feat_num_clear),
        ('Majority(Q\')', generator.feat_majority,),
        ('Ratio(Q\')', generator.feat_ratio),
        ('Fraction(Q\')', generator.feat_fraction),
        ('(ii) Features on CQ\'', blank),
        ('==================', blank),
    ]

    features_unclear = [
        ('CQGlobal(q,CQ\')', generator.feat_global_cos),
        ('CQIndividual(q,CQ\')', generator.feat_individual_cos),
        ('CQWeighted(q,CQ\')', generator.feat_individual_cos_weighted)
    ]

    print()
    for name, feature in feature_table:
        print('{} = {}'.format(name, feature(p_id, q_ids, q_scores)))

    for name, feature in features_unclear:
        print('{} = {}'.format(name, feature(p_id, q_ids=[1, 3], q_scores=[5, 1])))
