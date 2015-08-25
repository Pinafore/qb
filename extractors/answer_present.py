# -*- coding: utf-8 -*-
from math import log
import time

import fuzzywuzzy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
stop = stopwords.words('english')

from feature_extractor import FeatureExtractor


kNEG_INF = float("-inf")


class AnswerPresent(FeatureExtractor):
    @staticmethod
    def has_guess():
        return False

    def name(self):
        return "answer_present"

    def score_one_guess(self, title, text):
        d = {}
        if "(" in title:
            title = title[:title.find("(")].strip()
        val = fuzz.partial_ratio(title, text)
        d["raw"] = log(val + 1)
        d["length"] = log(val * len(title) / 100. + 1)

        longest_match = 1
        for ii in title.split():
            if ii.lower() in stop:
                continue
            longest_match = max(longest_match, len(ii) if ii in text else 0)
        d["longest"] = log(longest_match)

        return d

    def vw_from_title(self, title, text):
        val = self.score_one_guess(title, text)
        return self.vw_from_score(val)

    def vw_from_score(self, results):
        return "|%s %s" % (self.name(), " ".join("%s:%f" % (x, results[x])
                                                 for x in results))

if __name__ == "__main__":
    tests = {}

    tests[u'cole'] = u"""This painter's indulgence of visual fantasy, and appreciation of different
    historic architectural styles can be seen in his 1840 Architect's Dream.
    After a series of paintings on The Last of the Mohicans, he made a three
    year trip to Europe in 1829, but he is better known for a trip four years
    earlier in which he journeyed up the Hudson River to the Catskill
    Mountains. FTP, name this painter of The Oxbow and The Voyage of Life
    series."""

    tests[u'mohicans'] = u"""In this book, the French general Montcalm shows one character a letter from General Webb, which prompt ons a long march during which a massacre occurs. The beginning of this novel sees Colonel Munro's two daughters being escorted to Fort William Henry. The evil Magua kills Uncas and Cora only to be shot later by (*) Natty Bumppo in this work. For 10 points, what second novel of James Fenimore Cooper's Leatherstocking Tales is titled for the sole survivor of a Native American tribe?"""

    tests[u'cooper'] = u"""This author wrote a novel in which a character who enjoys "fish with dynamite" is the right hand man of a man who almost died by a falling tree and cherishes the view atop "Mount Vision", Judge Marmaduke Temple. In another of his works, Tamenund, the "Sache" of the Delaware, frees his prisoners after witnessing a turtle tattoo. In that work by this author of (*) The Pioneers, Magua kills the son of Chingachgook, Uncas, despite the attempts of Hawkeye to save him. For ten points, name this American author who wrote about Natty Bumppo in The Last of the Mohicans, the second of his Leatherstocking Tales."""

    tests[u'seward'] = u"""In 1850, this politician promoted the anti-slavery cause with a speech on the Senate floor appealing to "a higher law than the Constitution." Lewis Powell stabbed this man on the same night that John Wilkes Booth killed the president. As the book Team of Rivals documented, this New York Senator joined Abraham Lincoln's Cabinet despite initially being the front-runner at the 1860 Republican Convention. For 10 points, name this Secretary of State for both Lincoln and Andrew Johnson, a man whose namesake "folly" was his purchase of Alaska."""

    tests[u'alaska'] = u"""The editor of the Daily Morning Chronicle was given a thirty thousand dollar bribe by Edouard de Stoeckl to help secure that newspaper's support for this action. A year after assaulting fellow Congressman Josiah Bushnell Grinnell with a cane, Lovell Rousseau presided over the official ceremony marking this event. Lobbyist Robert J. Walker bribed Congressmen to approve of this action, as did Minister (*) Stoeckl. This action helped one party repay a 15 million pound loan to the Rothschilds. A holiday commemorating it falls on October 18th, not the October 7th on which it occurred in the Julian calendar. It cost 7.2 million dollars and was not popular until gold was discovered in the Klondike. For 10 points, name this action dubbed Seward's Folly."""

    start = time.time()
    print("Startup: %f sec" % (time.time() - start))


    guesses = ["Thomas Cole", "James Fenimore Cooper", "The Last of the Mohicans", "William H. Seward", "Alaska Purchase"]
    ge = AnswerPresent()

    for ii in tests:
        print("-------------------------------")
        print(ii)

        print("\n")
        for gg in guesses:
            start = time.time()
            print(gg, ge.vw_from_title(gg, tests[ii]))
            elapsed = time.time() - start
            print("%f secs" % (elapsed))
