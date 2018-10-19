# pylint: disable=no-member
import logging
from bson import json_util
from flask import Flask, Response, request
from flask_cors import CORS

from qac.storage import base, post_storage

app = Flask(__name__)
CORS(app)

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.DEBUG)

COMMUNITY_STORAGE = {}
DB = base.Storage()
PAGE_SIZE = 10


def jsonify(bson):
    return Response(
        json_util.dumps(bson),
        mimetype='application/json'
    )


def get_storage(community):
    cached_storage = COMMUNITY_STORAGE.get(community, None)
    if not cached_storage:
        cached_storage = post_storage.PostStorage(community)
        COMMUNITY_STORAGE[community] = cached_storage

    return cached_storage


@app.route('/communities')
def communities():
    return jsonify({'communities': DB.communities()})


@app.route('/<community>/questions')
def questions(community):
    page = request.args.get('page', default=1, type=int)
    label = request.args.get('label', default=None, type=str)

    query = {}
    if label:
        query['qac_annotation.label'] = {'$eq': label}

    if label == 'UNCLEAR':
        # only return unclear posts that have a clarifying answer
        query['$or'] = [
            {'qac_annotation.clarification_edit': {'$exists': True}},
            {'qac_annotation.clarification_comment': {'$exists': True}}
        ]

    question_list = get_storage(community).get_paginated(
        page=page, page_size=PAGE_SIZE, query=query
    )
    return jsonify({'questions': question_list})


@app.route('/<community>/questions/<post_id>')
def question(community, post_id):
    post = get_storage(community).get(post_id)
    return jsonify({'questions': [post]})


@app.route('/<community>/questions/<post_id>/clarq/categorization', methods=['PUT'])
def annotate_clarq(community, post_id):
    s = get_storage(community)
    data = json_util.loads(request.data.decode('utf-8')) # explicit decode for older pymongo
    app.logger.debug('Add annotation for post_id=%s in community=%s. Annotation=%s',
                     post_id, community, data)
    updated = s.annotate_clarification_question(
        post_id, data['user'], data['categories']
    )

    next_question = s.next_unclear(updated['_id'])
    app.logger.debug('Next unclear %s', next_question)
    return jsonify({'question': updated, 'next': next_question})

@app.route('/users', methods=['GET', 'POST'])
def users():
    if request.method == 'POST':
        data = json_util.loads(request.data.decode('utf-8')) # explicit decode for older pymongo
        app.logger.debug('Create new user=%s', data)
        result = DB.add_user(data['name'])
        return jsonify({'_id': result})

    return jsonify({'users': DB.users()})

if __name__ == '__main__':
    app.run(debug=True)
