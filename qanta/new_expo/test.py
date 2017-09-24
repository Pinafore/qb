import random
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.new_expo.game import Game
from qanta.new_expo.agent import HumanAgent, GuesserBuzzerAgent

class StupidBuzzer:
    
    def __init__(self, threshold=1.2):
        self.threshold = threshold

    def new_round(self):
        pass

    def buzz(self, guesses):
        '''guesses is a sorted list of (guess, score)'''
        return guesses[0][1] > self.threshold

class GuesserWrapper:

    def __init__(self, guesser):
        self.guesser = guesser

    def new_round(self):
        pass

    def guess(self, text):
        return self.guesser.guess_single(text)

def main():
    # setup questions
    questions = list(QuestionDatabase().all_questions().values())
    dev_questions = [x for x in questions if x.fold == 'dev']
    random.shuffle(dev_questions)

    # setup human agent
    human_agent = HumanAgent()

    # setup machine agent
    gspec = AbstractGuesser.list_enabled_guessers()[0]
    guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
            gspec.guesser_class, '')
    guesser = ElasticSearchWikidataGuesser.load(guesser_dir)
    guesser = GuesserWrapper(guesser)
    buzzer = StupidBuzzer(threshold=1.2)
    machine_agent = GuesserBuzzerAgent(guesser, buzzer)

    # setup game
    game = Game(dev_questions, [human_agent, machine_agent], [])

    game.run(3)

if __name__ == '__main__':
    main()
