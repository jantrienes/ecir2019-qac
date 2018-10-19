import re
from nltk import tokenize

TOKENIZER = tokenize.RegexpTokenizer(r'[\w\'\-]+|(?:[\.,\/#!$\"\?%\^&\*;:{}=\-_`~()\[\]])')
URL_REGEX = re.compile(r'(https?:\/\/[^ )]+)', re.MULTILINE)

def _clean(text):
    # match source code (lines starting with 4 empty spaces)
    text_cleaned = re.sub(r'^\ {4,}.*', '', text, flags=re.MULTILINE)
    text_cleaned = text_cleaned.replace('\n', '')
    text_cleaned = text_cleaned.replace('\r', '')
    return text_cleaned

def _tokens(text):
    tokens = TOKENIZER.tokenize(text)
    filtered = [token for token in tokens if token.isalpha()]
    return filtered

class ReadabilityAnalyzer(object):

    def __init__(self, question):
        self.raw = URL_REGEX.sub('URL', question)
        self.cleaned = _clean(self.raw)

    @property
    def tokens(self):
        return _tokens(self.cleaned)

    @property
    def sents(self):
        return tokenize.sent_tokenize(self.cleaned)

    @property
    def coleman_liau_index(self):
        chars, words, sents = len(self.cleaned), len(self.tokens), len(self.sents)
        if words == 0:
            return 0
        return (5.89 * chars / words) - (30.0 * (sents / words)) - 15.8
