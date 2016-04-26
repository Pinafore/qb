import unittest
from reporting.generate_buzzes import load_meta, load_predictions, Meta, Prediction


class TestGenerateBuzzes(unittest.TestCase):
    def test_load_meta(self):
        meta = load_meta('test/data/test.meta').list()
        expect = [
            Meta(171314, 2, 0, 'A Season in Hell'),
            Meta(175857, 3, 0, 'Battle of Poltava'),
            Meta(188507, 5, 0, 'Leonhard Euler'),
            Meta(168558, 4, 0, 'Nigeria'),
            Meta(177292, 2, 0, 'Lee Harvey Oswald'),
            Meta(166027, 3, 0, 'Booker T. Washington'),
            Meta(146308, 1, 0, 'Carrying capacity'),
            Meta(163629, 1, 0, 'Zimmermann Telegram'),
            Meta(176563, 3, 0, 'Ivan Turgenev'),
            Meta(179316, 3, 0, 'Kennedy family'),
        ]
        self.assertListEqual(meta, expect)

    def test_load_predictions(self):
        predictions = load_predictions('test/data/test.pred').list()
        expect = [
            Prediction(score=-11.75036, question=171314, sentence=2, token=0),
            Prediction(score=-10.513688, question=175857, sentence=3, token=0),
            Prediction(score=-2.096656, question=188507, sentence=5, token=0),
            Prediction(score=-7.034301, question=168558, sentence=4, token=0),
            Prediction(score=-12.733343, question=177292, sentence=2, token=0),
            Prediction(score=-8.907219, question=166027, sentence=3, token=0),
            Prediction(score=-7.256502, question=146308, sentence=1, token=0),
            Prediction(score=-9.449733, question=163629, sentence=1, token=0),
            Prediction(score=-9.682533, question=176563, sentence=3, token=0),
            Prediction(score=-10.097745, question=179316, sentence=3, token=0)
        ]
        self.assertListEqual(predictions, expect)
