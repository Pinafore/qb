from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.search.whoosh import WhooshWikiIndex


class IrExtractor(AbstractFeatureExtractor):
    def __init__(self):
        super(IrExtractor, self).__init__()
        self.wiki_index = WhooshWikiIndex()

    @property
    def name(self):
        return 'ir'

    def score_guesses(self, guesses, text):
        pass

    def vw_from_score(self, results):
        pass

    def text_guess(self, text):
        return dict(self.wiki_index.search(text))



