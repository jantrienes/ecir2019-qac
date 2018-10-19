# pylint: disable=W0212
from qac.dataset import annotation
from qac.dataset.annotation import (ClarificationQuestion, parse_time)

annotator = annotation.Annotator()

def test_get_question():
    question = annotation.get_question("Include this. What is \"caulk\"? Remove this.")
    assert question == 'include this . what is " caulk " ?'

    question = annotation.get_question("this comment has no question!")
    assert question is None

    question = annotation.get_question("Have you checked [this](https://website.com?id=123)?")
    assert question == "have you checked [ this ] ( url ) ?"

    question = annotation.get_question("Have you checked this: https://website.com?id=123 ?")
    assert question == "have you checked this : url ?"

    question = annotation.get_question("?")
    assert question == "?"


def test_clear():
    question = {
        "Id": "1",
        "Edits": [{}, {}, {}, {}],
        "Comments": [{}]
    }

    assert annotator.clear(question) is False
    question['Edits'] = [{}, {}, {}]
    assert annotator.clear(question) is False
    question.pop('Comments')
    assert annotator.clear(question) is False
    question['AcceptedAnswerId'] = 1
    assert annotator.clear(question) is True


def test_unclear():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this is a question?",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000"
        }]
    }

    assert annotator.unclear(question) is True


def test_unclear_multiple_questions():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this is a question?",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000"
        }, {
            "Id": "2",
            "Text": "is this is another question?",
            "UserId": "3",
            "CreationDate": "2010-07-22T08:00:00.000"
        }, {
            "Id": "3",
            "Text": "owner asks?",
            "UserId": "1",
            "CreationDate": "2010-07-22T09:00:00.000"
        }]
    }

    assert annotator.unclear(question) is True


def test_clarifying_question():
    c1 = {
        "Id": "1",
        "Text": "is this a question? some unrelated text.",
        "UserId": "3",
        "CreationDate": "2010-07-22T07:00:00.000",
    }

    c2 = {
        "Id": "2",
        "Text": "is this is another question?",
        "UserId": "2",
        "CreationDate": "2010-07-22T08:00:00.000",
    }

    question = {
        "OwnerUserId": "1",
        # in wrong order, assert method sorts by timestamp
        "Comments": [c2, c1]
    }

    cq = annotator.clarification_question(question)
    assert cq.text == "is this a question ?"
    assert cq.creation_date == parse_time("2010-07-22T07:00:00.000")


def test_clarification_comment():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }, {
            "Id": "2",
            "Text": "yes, this is indeed a question!",
            "UserId": "1",
            "CreationDate": "2010-07-22T08:00:00.000",
        }, {
            "Id": "3",
            "Text": "another comment!",
            "UserId": "1",
            "CreationDate": "2010-07-22T08:01:00.000",
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"),
                               entity_id="1")
    cc = annotator._clarification_comment(question, cq)

    assert cc.entity_id == "2"
    assert cc.text == 'yes , this is indeed a question !'
    assert cc.creation_date == parse_time("2010-07-22T08:00:00.000")


def test_clarification_comment_ignore_previous_comments():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "comment before question",
            "UserId": "1",
            "CreationDate": "2010-07-22T06:00:00.000",
        }, {
            "Id": "2",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="2")
    cc = annotator._clarification_comment(question, cq)

    assert cc is None


def test_clarification_comment_not_present():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="1")
    cc = annotator._clarification_comment(question, cq)

    assert cc is None

    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }, {
            "Id": "2",
            "Text": "unrelated comment by unrelated user",
            "UserId": "3",
            "CreationDate": "2010-07-22T08:00:00.000",
        }]
    }

    cc = annotator._clarification_comment(question, cq)
    assert cc is None


initial_edits = [{
    "Id": "1",
    "PostHistoryTypeId": "2",  # initial body
    "RevisionGUID": "guid1",
    "CreationDate": "2010-07-22T06:00:00.000",
    "UserId": "1",
    "Text": "question body"
}, {
    "Id": "2",
    "PostHistoryTypeId": "1",  # initial title
    "RevisionGUID": "guid1",
    "CreationDate": "2010-07-22T06:00:00.000",
    "UserId": "1",
    "Text": "title"
}, {
    "Id": "3",
    "PostHistoryTypeId": "3",  # initial tags
    "RevisionGUID": "guid1",
    "CreationDate": "2010-07-22T06:00:00.000",
    "UserId": "1",
    "Text": "<tag1><tag2>"
}]


def test_clarification_edit():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }],
        "Edits": initial_edits + [{
            "Id": "4",
            "PostHistoryTypeId": "4",  # title updated
            "RevisionGUID": "guid2",
            "CreationDate": "2010-07-22T08:00:00.000",
            "UserId": "1",
            "Text": "title updated"
        }, {
            "Id": "5",
            "PostHistoryTypeId": "5",  # edit body
            "RevisionGUID": "guid3",
            "CreationDate": "2010-07-22T08:00:00.000",
            "UserId": "1",
            "Text": "question body, answer to question"
        }, {
            "Id": "6",
            "PostHistoryTypeId": "5",  # edit body
            "RevisionGUID": "guid4",
            "CreationDate": "2010-07-22T09:00:00.000",
            "UserId": "1",
            "Text": "question body, answer to question, more information"
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="1")
    ce = annotator._clarification_edit(question, cq)

    assert ce.entity_id == "5"
    assert ce.text == "question body , answer to question"
    assert ce.creation_date == parse_time("2010-07-22T08:00:00.000")

def test_clarification_edit_diff():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }],
        "Edits": initial_edits + [{
            "Id": "5",
            "PostHistoryTypeId": "5",  # edit body
            "RevisionGUID": "guid3",
            "CreationDate": "2010-07-22T08:00:00.000",
            "UserId": "1",
            "Text": "question body, answer to question"
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="1")
    ce = annotator._clarification_edit(question, cq)

    assert ce.text == "question body , answer to question"
    assert ce.diff.insert == " , answer to question"
    assert ce.diff.start_offset == 13
    assert ce.diff.end_offset == 34

def test_clarification_edit_initial_post():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }],
        "Edits": initial_edits
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="1")
    ce = annotator._clarification_edit(question, cq)
    assert ce is None


def test_clarification():
    question = {
        "OwnerUserId": "1",
        "Comments": [{
            "Id": "1",
            "Text": "is this a question? some unrelated text.",
            "UserId": "2",
            "CreationDate": "2010-07-22T07:00:00.000",
        }, {
            "Id": "2",
            "Text": "yes this is my question",
            "UserId": "1",
            "CreationDate": "2010-07-22T08:00:00.000",
        }],
        "Edits": initial_edits + [{
            "Id": "4",
            "PostHistoryTypeId": "5",  # edit body
            "RevisionGUID": "guid3",
            "CreationDate": "2010-07-22T08:00:00.000",
            "UserId": "1",
            "Text": "question body, answer to question"
        }]
    }

    cq = ClarificationQuestion("is this a question ?",
                               parse_time("2010-07-22T07:00:00.000"), entity_id="1")
    clarification_comment, clarification_edit = annotator.clarification(question, cq)
    assert clarification_comment.text == "yes this is my question"
    assert clarification_edit.diff.insert == " , answer to question"
