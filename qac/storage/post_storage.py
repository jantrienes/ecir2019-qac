"""MongoDB API for storing and retrieving post objects.

The Stack Exchange dump is denormalized: a post is an object where comments and edits are properties
of that post.
"""
import logging

import pymongo

from qac.dataset.annotation import QuestionLabels
from qac.storage.base import Storage

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def _entity_dict(entity):
    return {
        'Id': entity.entity_id,
        'Text': entity.text,
        'CreationDate': entity.creation_date
    }


class PostStorage(Storage):

    def __init__(self, community, host=None, port=None, db=None):
        """Initialize Mongo connection.

        :param community: suffix of posts collection.
        """
        super(PostStorage, self).__init__(host, port, db)
        self._collection = self._db['posts_{}'.format(community)]

    def save_post(self, post):
        return self._collection.insert_one(post)

    def _push(self, post_id, prop_name, obj):
        """Adds an object to array property of post with given ID.

        If the list does not exist, it is created."""
        return self._collection.update_one({'Id': post_id}, {"$push": {prop_name: obj}})

    def add_comment(self, post_id, comment):
        return self._push(post_id, 'Comments', comment)

    def add_edit(self, post_id, edit):
        return self._push(post_id, 'Edits', edit)

    def get(self, post_id):
        return self._collection.find_one({'Id': {'$eq': post_id}})

    def get_all(self):
        return self.find({})

    def find(self, query):
        return self._collection.find(query)

    def get_paginated(self, page, page_size, query={}):
        cursor = self._collection \
            .find(query) \
            .sort([('_id', pymongo.ASCENDING)]) \
            .skip(page * page_size) \
            .limit(page_size)

        return [q for q in cursor]

    def count(self):
        return self._collection.count({})

    def create_id_index(self):
        """Indexes the Id field of the posts collection to allow for efficient querying."""
        return super(PostStorage, self).create_index(self._collection.name, ['Id'])

    def add_annotation(self, post_id, label, question, clarification_comment, clarification_edit):
        obj = {
            'label': label.name,
        }

        if question:
            obj['question'] = _entity_dict(question)

        if clarification_comment:
            obj['clarification_comment'] = _entity_dict(clarification_comment)

        if clarification_edit and clarification_edit.diff:
            edit_dict = _entity_dict(clarification_edit)
            edit_dict['InitialText'] = clarification_edit.initial_text
            edit_dict['Insert'] = clarification_edit.diff.insert
            edit_dict['StartOffset'] = clarification_edit.diff.start_offset
            edit_dict['EndOffset'] = clarification_edit.diff.end_offset
            obj['clarification_edit'] = edit_dict

        return self._collection.update_one({'Id': post_id}, {"$set": {'qac_annotation': obj}})

    def annotation_statistics(self):
        stats = {}
        stats['total_questions'] = self.count()
        labels = [l.name for l in QuestionLabels]

        for label in labels:
            stats['total_{}'.format(label)] = self._collection.count(
                {'qac_annotation.label': label}
            )

        lbl_unclear = QuestionLabels.UNCLEAR.name
        stats['total_{}_unanswered'.format(lbl_unclear)] = self._collection.count({
            'qac_annotation.label': QuestionLabels.UNCLEAR.name,
            'qac_annotation.clarification_comment': {'$exists': False},
            'qac_annotation.clarification_edit': {'$exists': False},
        })
        stats['total_{}_answered_comment'.format(lbl_unclear)] = self._collection.count({
            'qac_annotation.label': QuestionLabels.UNCLEAR.name,
            'qac_annotation.clarification_comment': {'$exists': True},
            'qac_annotation.clarification_edit': {'$exists': False},
        })
        stats['total_{}_answered_edit'.format(lbl_unclear)] = self._collection.count({
            'qac_annotation.label': QuestionLabels.UNCLEAR.name,
            'qac_annotation.clarification_comment': {'$exists': False},
            'qac_annotation.clarification_edit': {'$exists': True},
        })
        stats['total_{}_answered_both'.format(lbl_unclear)] = self._collection.count({
            'qac_annotation.label': QuestionLabels.UNCLEAR.name,
            'qac_annotation.clarification_comment': {'$exists': True},
            'qac_annotation.clarification_edit': {'$exists': True},
        })

        return stats

    def annotate_clarification_question(self, post_id, user, categories):
        query = {
            'Id': {'$eq': post_id},
            'qac_annotation.question.category_annotations.user._id': {'$eq': user['_id']}
        }

        if self._collection.find_one(query) is not None:
            return self._collection.find_one_and_update(query, {
                '$set': {'qac_annotation.question.category_annotations.$.categories': categories}
            }, return_document=pymongo.ReturnDocument.AFTER)

        return self._collection.find_one_and_update(
            {'Id': post_id},
            {"$push": {
                'qac_annotation.question.category_annotations': {
                    'user': user,
                    'categories': categories
                    }
                }
            },
            return_document=pymongo.ReturnDocument.AFTER)

    def next_unclear(self, previous_mongo_id):
        query = {}
        query['_id'] = {'$gt': previous_mongo_id}
        query['qac_annotation.label'] = {'$eq': QuestionLabels.UNCLEAR.name}
        query['$or'] = [
            {'qac_annotation.clarification_edit': {'$exists': True}},
            {'qac_annotation.clarification_comment': {'$exists': True}}
        ]

        return self._collection.find_one(query, { 'Id': 1 })
