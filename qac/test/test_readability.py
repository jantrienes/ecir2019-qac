import pytest

from qac.experiments import readability


def test_readability_analyzer():
    question = "This is my question which has two sentences. This is the second sentence?"
    analyzer = readability.ReadabilityAnalyzer(question)
    assert analyzer.coleman_liau_index == pytest.approx(12.659, rel=1e-3)
