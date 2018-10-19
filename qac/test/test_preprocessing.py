from qac.experiments import preprocessing

def test_pad_sequence():
    seq = ['A', 'B', 'C']
    assert preprocessing.pad_sequence(seq, max_length=3) == seq
    assert preprocessing.pad_sequence(seq, max_length=2) == ['A', 'B']
    assert preprocessing.pad_sequence(seq, max_length=4) == ['A', 'B', 'C', '<PAD>']
