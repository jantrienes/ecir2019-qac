import difflib
import re
from collections import namedtuple
from datetime import datetime

from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer(r'[\w\'\-]+|(?:[\.,\/#!$\"\?%\^&\*;:{}=\-_`~()\[\]])')
URL_REGEX = re.compile(r"(https?:\/\/[^ )]+)", re.MULTILINE)


def get_tokens(text):
    return TOKENIZER.tokenize(text.lower())


def remove_urls(text):
    return URL_REGEX.sub("URL", text)


def parse_time(time_string):
    return datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")


def format_time(struct_time):
    return datetime.strftime(struct_time, "%Y-%m-%dT%H:%M:%S.%f")[:-3]


def timestamp_sorted(lst):
    return sorted(lst, key=lambda e: parse_time(e['CreationDate']))


Diff = namedtuple('Diff', ['insert', 'start_offset', 'end_offset'])


def get_diff(initial, final):
    """Compare two strings and extract last inserted string.

    Adapted from:
    https://github.com/raosudha89/ranking_clarification_questions/blob/f1cadac67915ece0756225086ccf6d553ff90905/src/data_generation/post_ques_ans_generator.py#L21

    As used in:
    Rao, S., & Daumé III, H. (2018). Learning to Ask Good Questions: Ranking Clarification
    Questions using Neural Expected Value of Perfect Information. arXiv preprint arXiv:1805.04655.

    :return: tuple(insert, start_offset, end_offset)
    """
    s = difflib.SequenceMatcher(None, initial, final)
    insert, start_offset, end_offset = None, None, None

    for tag, _, _, start_offset, end_offset in s.get_opcodes():
        if tag == 'insert':
            insert = final[start_offset:end_offset]
    if not insert:
        return None
    return Diff(insert, start_offset, end_offset)


def print_statistics(stats):
    def _print_value(field, stats):
        print('{0: <34} {1: <12}'.format(field, str(stats[field])))

    def _print_value_fraction(field, total, stats):
        print('{0: <34} {1: <12} {2:.2f}'.format(field, stats[field], stats[field] / total))

    total = stats['total_questions']
    total_unclear = stats['total_UNCLEAR']

    print('{0: <34} {1: <12} {2:}'.format('Category', 'Instances', 'Frac'))
    print('{}'.format('-' * 52))
    _print_value('total_questions', stats)
    _print_value_fraction('total_CLEAR', total, stats)
    _print_value_fraction('total_OTHER', total, stats)
    _print_value_fraction('total_UNCLEAR', total, stats)
    _print_value_fraction('total_UNCLEAR_unanswered', total_unclear, stats)
    _print_value_fraction('total_UNCLEAR_answered_edit', total_unclear, stats)
    _print_value_fraction('total_UNCLEAR_answered_comment', total_unclear, stats)
    _print_value_fraction('total_UNCLEAR_answered_both', total_unclear, stats)
