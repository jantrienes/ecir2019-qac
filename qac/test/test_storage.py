# pylint: disable=protected-access
from qac.storage import base


class TestStorage(object):
    storage = base.Storage(db='stackexchange_test')

    @classmethod
    def setup_class(cls):
        cls.storage._db['users'].delete_many({})

    @classmethod
    def teardown_class(cls):
        cls.storage._db['users'].delete_many({})


    def test_users(self):
        assert not list(self.storage.users())

        self.storage.add_user('john')
        self.storage.add_user('jane')

        users = list(self.storage.users())
        assert len(users) == 2
        assert len(list(filter(lambda u: u['name'] == 'john', users))) == 1
        assert len(list(filter(lambda u: u['name'] == 'jane', users))) == 1
