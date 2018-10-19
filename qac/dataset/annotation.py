# pylint: disable=R0903, C0321
from enum import Enum

from qac.dataset.util import (get_diff, get_tokens, parse_time, remove_urls,
                              timestamp_sorted)


class QuestionLabels(Enum):
    CLEAR = 'CLEAR'
    UNCLEAR = 'UNCLEAR'
    OTHER = 'OTHER'


class Entity(object):

    def __init__(self, text, creation_date, entity_id):
        self.entity_id = entity_id
        self.text = text
        self.creation_date = creation_date


class ClarificationQuestion(Entity):
    pass


class ClarificationComment(Entity):
    pass


class ClarificationEdit(Entity):

    def __init__(self, text, creation_date, entity_id, initial_text):
        super(ClarificationEdit, self).__init__(text, creation_date, entity_id)
        self.initial_text = initial_text
        self.diff = get_diff(self.initial_text, self.text)


def preprocess(text):
    text = remove_urls(text)
    return get_tokens(text)


def get_question(text):
    tokens = preprocess(text)

    try:
        i = tokens.index('?')
        return ' '.join(tokens[:i + 1])
    except ValueError:
        return None


class Annotator(object):

    def clear(self, post):
        # Every question has in its initial revision 3 edits (initial title/body/tags).
        if len(post['Edits']) == 3 and 'Comments' not in post and 'AcceptedAnswerId' in post:
            return True
        return False

    def unclear(self, post):
        if self.clarification_question(post):
            return True

        return False

    def assign_label(self, post):
        if self.clear(post):
            return QuestionLabels.CLEAR

        if self.unclear(post):
            return QuestionLabels.UNCLEAR

        return QuestionLabels.OTHER

    def clarification_question(self, post):
        """Extract first clarification question from the comments of given post.

        :return: A ClarificationQuestion if present and None otherwise.
        """
        if 'Comments' not in post:
            return None

        question = None
        for comment in timestamp_sorted(post['Comments']):
            question_text = get_question(comment['Text'])
            if question_text and post.get('OwnerUserId', None) != comment.get('UserId', None):
                question = ClarificationQuestion(question_text,
                                                 parse_time(comment['CreationDate']),
                                                 comment['Id'])
                break

        return question

    def _clarification_comment(self, post, clarification_question):
        """Extract first comment by the original post author in response to the given clarification
        question.

        :return: ClarificationComment if present and None otherwise.
        """
        clarifying_comment = None
        owner_id = post.get('OwnerUserId', None)

        for comment in timestamp_sorted(post['Comments']):
            if comment.get('UserId', None) != owner_id:
                continue

            comment_time = parse_time(comment['CreationDate'])
            if comment_time > clarification_question.creation_date:
                tokens = preprocess(comment['Text'])
                clarifying_comment = ClarificationComment(
                    ' '.join(tokens), comment_time, comment['Id'])
                break

        return clarifying_comment

    def _initial_revision(self, post):
        for edit in post['Edits']:
            if int(edit['PostHistoryTypeId']) == 2:
                tokens = preprocess(edit['Text'])
                return ' '.join(tokens)

        return None

    def _clarification_edit(self, post, clarification_question):
        """Extract first edit by the original post author in response to the given clarification
        question.

        :return: ClarificationEdit if present and None otherwise.
        """
        clarification_edit = None
        owner_id = post.get('OwnerUserId', None)

        initial_text = self._initial_revision(post)

        for edit in timestamp_sorted(post['Edits']):
            if int(edit['PostHistoryTypeId']) != 5:  # only consider question body updates
                continue
            if edit.get('UserId', None) != owner_id:
                continue

            edit_time = parse_time(edit['CreationDate'])
            if edit_time > clarification_question.creation_date:
                tokens = preprocess(edit['Text'])
                clarification_edit = ClarificationEdit(' '.join(tokens), edit_time,
                                                       edit['Id'], initial_text)
                break

        return clarification_edit

    def clarification(self, post, clarification_question):
        clarification_comment = self._clarification_comment(post, clarification_question)
        clarification_edit = self._clarification_edit(post, clarification_question)

        return clarification_comment, clarification_edit
