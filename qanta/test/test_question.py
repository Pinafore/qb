import unittest
from qanta.datasets.quiz_bowl import Question


class TestQuestion(unittest.TestCase):
    def test_partial_sentences(self):
        q = Question(0, '', None, False, '', 'Thomas Cole', None, None, None)
        first_sentence = "This painter's indulgence of visual fantasy, and appreciation of " \
                         "different historic architectural styles can be seen in his 1840 " \
                         "Architect's Dream."
        second_sentence = 'After a series of paintings on The Last of the Mohicans, he made a ' \
                          'three year trip to Europe in 1829, but he is better known for a trip ' \
                          'four years earlier in which he journeyed up the Hudson River to the ' \
                          'Catskill Mountains.'
        third_sentence = 'FTP, name this painter of The Oxbow and The Voyage of Life series.'
        q.add_text(0, first_sentence)
        q.add_text(1, second_sentence)
        q.add_text(2, third_sentence)
        sentence_partials = list(q.partials())
        self.assertEqual(len(sentence_partials), 3)
        self.assertEqual(sentence_partials[0][0], 1)
        self.assertEqual(sentence_partials[0][1], 0)
        self.assertEqual(sentence_partials[0][2][0], first_sentence)

        self.assertEqual(sentence_partials[1][0], 2)
        self.assertEqual(sentence_partials[1][1], 0)
        self.assertEqual(sentence_partials[1][2][0], first_sentence)
        self.assertEqual(sentence_partials[1][2][1], second_sentence)

        self.assertEqual(sentence_partials[2][0], 3)
        self.assertEqual(sentence_partials[2][1], 0)
        self.assertEqual(sentence_partials[2][2][0], first_sentence)
        self.assertEqual(sentence_partials[2][2][1], second_sentence)
        self.assertEqual(sentence_partials[2][2][2], third_sentence)

        word_partials = list(q.partials(word_skip=1))
        self.assertEqual(len(word_partials), 78)
        self.assertEqual(word_partials[0][2][0], 'This')
        self.assertEqual(len(word_partials[0][2]), 1)
