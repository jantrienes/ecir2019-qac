# pylint: disable=protected-access
from qac.storage import post_storage


class TestStorage(object):
    storage = post_storage.PostStorage(community='test', db='stackexchange_test')

    @classmethod
    def setup_class(cls):
        cls.storage.save_post({
            'Id': '1',
            "qac_annotation": {
                "label": "UNCLEAR",
                "question": {
                    "Id": "1",
                    "category_annotations": [
                        {
                            "user": {'_id': '1'},
                            "categories": [
                                "A",
                                "B",
                                "C"
                            ]
                        },
                    ]
                }
            }
        })

    @classmethod
    def teardown_class(cls):
        cls.storage._collection.delete_many({})

    def test_annotate_clarification_question(self):
        def get_annotations(post_id):
            q = self.storage.get(post_id)
            return q['qac_annotation']['question']['category_annotations']

        assert len(get_annotations('1')) == 1

        self.storage.annotate_clarification_question('1', {'_id': '1'}, ['A'])
        annotations = get_annotations('1')
        assert len(annotations) == 1
        assert annotations[0]['categories'] == ['A']

        self.storage.annotate_clarification_question('1', {'_id': '2'}, ['A', 'B', 'C'])
        annotations = get_annotations('1')
        assert len(annotations) == 2

        user1 = [a for a in annotations if a['user']['_id'] == '1'][0]
        user2 = [a for a in annotations if a['user']['_id'] == '2'][0]

        assert user1['categories'] == ['A']
        assert user2['categories'] == ['A', 'B', 'C']
