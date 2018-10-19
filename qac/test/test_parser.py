import os
from qac.dataset import parser

def test_parser():
    path = os.path.dirname(__file__)
    test_file = os.path.join(path, 'test_parser.xml')

    posts_parser = parser.XMLParser(test_file)

    posts = [post for post in posts_parser]
    assert len(posts) == 2
    p1 = posts[0]
    assert p1.Id == '1'
    assert p1.PostTypeId == '1'
    assert p1.Body == 'Post 1'
