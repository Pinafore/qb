
# -*- coding: utf-8 -*-
import sqlite3

from nltk import pos_tag
from collections import defaultdict
from feature_extractor import FeatureExtractor


kNEG_INF = float("-inf")
class AnsPresent(FeatureExtractor):
    def __init__(self, gender_db = 'data/questions.db', gender_table = 'questions'):
        self._gender_table = gender_table
        self._features = ["title_not_present"]
        self.conn = sqlite3.connect(gender_db)
        self.cur = self.conn.cursor()
        self._name = self.name()
        self._gender_dict = self.build_gender_dict()
        self._title_nouns_dict = {}
        self.get_title_nouns()

    @staticmethod
    def has_guess():
        return True

    def name(self):
        return "ans_present"

    def get_title_nouns(self):
        for title in self._gender_dict:
            title_split = title.split()
            title_nouns = [x[0] for x in pos_tag(title_split) if x[1] in ['NN', 'NNPS', 'NNP', 'NNS']]
            self._title_nouns_dict.update({title: title_nouns})

    def build_gender_dict(self):
        query  = 'SELECT distinct page, gender FROM {};'.format(self._gender_table)
        c = self.cur.execute(query,)
        res = c.fetchall()
        return dict(res)

    # titles that are not present in question text will have match as 1.
    # to determine whether title is present in qtext, \
    # see if the title is a person and look for their last name in qtext
    # if not a person, look for nouns of title in qtext
    def score_one_guess(self, title, text):
        title_lower = title.lower()
        title_split = title_lower.split()
        text = text.lower()
        val = {}

        try:
            queried_gender = self._gender_dict[title]
            if queried_gender == "male":
                male_title = True
                female_title = False
            elif queried_gender == "female":
                female_title = True
                male_title = False
            else:
                male_title = False
                female_title = False
        except KeyError:
            male_title = False
            female_title = False


        for feature in self._features:
            if feature == "title_not_present":
                val[feature] = 0.
                if (male_title or female_title):
                    if len(title_split[-1]) > 4:
                        if title_split[-1] not in text:
                            val[feature] = 1.
                    else:
                        if (title_split[-1] + ' ') not in text:
                            val[feature] = 1.

                else:
                    val[feature] = 1.
                    if len(title_split) > 1:
                        try:
                            title_nouns = self._title_nouns_dict[title]
                            for noun in title_nouns:
                                noun = noun.lower()
                                if len(noun) > 4:
                                    if noun in text:
                                        val[feature] = 0.
                                else:
                                    if (noun + ' ') in text:
                                        val[feature] = 0.
                        except KeyError:
                            val[feature] = 1.
                    else:
                        if title_lower in text:
                            val[feature] = 0.


        return val


    def text_guess(self, text, ir_titles):
        res = defaultdict(dict)
        for title in ir_titles:
            res[title] = self.score_one_guess(title, text)

        return res

    def vw_from_title(self, title, text):
        val = self.score_one_guess(title, text)
        return self.vw_from_score(val)

    def vw_from_score(self, results):
        res = "|%s" % self._name
        for ii in results:
            if results[ii] > kNEG_INF:
                res += " %sfound:1 %sscore:%f" % \
                    (ii, ii, results[ii])
            else:
                res += " %sfound:0 %sscore:0.0" % (ii, ii)
        return res


if __name__ == "__main__":
    import time, argparse
    parser = argparse.ArgumentParser(description= "Checks if Title is present in Question Text")
    parser.add_argument("--gender_db", default = "data/questions.db", help = \
    "Name of db containing gender information of titles")
    parser.add_argument("--gender_table", default = "questions", help = \
    "Name of table in db containing gender information of titles")


    flags = parser.parse_args()

    tests = {}
    tests[u"A"] = u"""Cole sought out the pope to
    seek forgiveness of her sins, only to be told that just as the pope's staff
    would never (*) blossom, his sins are never be forgiven. Three days later,
    the pope's staff miraculously bore flowers. For 10 points, identify this
    politician from london the subject of an opera by Wagner [VAHG-ner]."""

    tests[u'B'] = u"""This painter's indulgence of visual fantasy, and appreciation
    of different historic architectural styles can be seen in his 1840 Architect's Dream.
    After a series of paintings on The Last of the Mohicans, he made a three year trip to
    Europe in 1829, but he is better known for a trip four years earlier in which he journeyed
    up the Hudson River to the Catskill Mountains. FTP, name this painter of The Oxbow and The
    Voyage of Life series."""

    tests[u'C'] = u"""Velvets gave way to brocades and silks and curie.  Darker colors gave way to blues,
    pinks, and whites.  War and religion gave way to a dream world where landscapes always depicted summer;
    where people flirted or made love; and where flowers, seaweed, and shells were all over the place.
    Indeed, the name for the movement comes from the French for ""shell.""
    For 10 points, name this 18th century art period, a late phase of, yet somewhat different from, the Baroque."""


    tests[u'D'] = u"""This tiny 1767 work in rococo is officially entitled ""The Happy Hazards of"" the title object.
    It was commissioned by the Baron de-Saint Julien, who asked to be painted in a position to see his mistress' legs,
    ""and even more of her if you wish to enliven your picture.""  The mistress is at center,
    kicking off her left shoe and being pushed by a smiling bishop.  FTP name this rococo garden scene,
    a masterpiece of Jean-Honore Fragonard."""

    tests[u'E'] = u"""The name, or rather nickname, is the same.  One was a 16th century musician and composer
    , a product of the Cretan Renaissance who was music master to the Duke of Bavaria and the cantor of Venice.
    The other was a painter who set eerie casts on the landscapes that he painted as a wanderer before settling
    in Toledo.  FTP, Give the shared nickname of Frangiskos Leondaritis and Dominikos Theotokopoulos which refers
    to their shared Cretan heritage."""


    tests[u'F'] = u"""An extramarital affair partially prompted his critique of the conflict between eroticism
    and religion in the essay ""Religious Rejections of the World and Their Directions"". The emotional
    domination exerted on him by his father led both to his 1898 mental breakdown and his advocacy of
    ""Liberal Imperialism"" to solve Germany's agrarian problems in his famous Freiburg Address.
    He gained fame for his analysis of German political and economic life in Economy and Society,
    but is more famous today for his investigation of the roots of the ""spirit of capitalism"".
    FTP, who was this German thinker, author of The Protestant in London?"""

    start = time.time()
    print("Startup: %f sec" % (time.time() - start))


    guesses = ["Thomas Cole", "Rococo", "Aubrey Beardsley", "London Bridge"]
    ge = TitleNotinQTextExtractor()

    for ii in tests:
        print(ii)
        start = time.time()

        res = dict(ge.text_guess(tests[ii], guesses))

        elapsed = time.time() - start

        print("%f secs for %s: %s" % (elapsed, ii, str(res)))
        print("\n")
        for gg in guesses:
            print("Score for %s: is"
            %gg, ge.score_one_guess(gg, tests[ii]))
