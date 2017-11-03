import random
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
from qanta.config import conf
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.new_expo.game import Game
from qanta.new_expo.agent import HumanAgent, GuesserBuzzerAgent, ESGuesserWrapper
from qanta.new_expo.agent import RNNBuzzer
from qanta.new_expo import hook

def test_buzzer():
    questions = QuestionDatabase().all_questions()
    buzzer = RNNBuzzer(model_dir='output/buzzer/neo_0.npz',
            word_skip=conf['buzzer_word_skip'])

    # setup machine agent
    gspec = AbstractGuesser.list_enabled_guessers()[0]
    guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
            gspec.guesser_class, '')
    guesser = ElasticSearchWikidataGuesser.load(guesser_dir)
    guesser = ESGuesserWrapper(guesser)

    key = list(questions.keys())[4]
    question = questions[key].flatten_text().split()
    for i, word in enumerate(question):
        clue = ' '.join(question[:i])
        guesses = guesser.guess(clue)
        buzz = buzzer.buzz(guesses)
        print(buzz)

def main():
    buzzer = RNNBuzzer()

    # setup questions
    questions = list(QuestionDatabase().all_questions().values())
    dev_questions = [x for x in questions if x.fold == 'dev']

    # setup machine agent
    gspec = AbstractGuesser.list_enabled_guessers()[0]
    guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
            gspec.guesser_class, '')
    guesser = ElasticSearchWikidataGuesser.load(guesser_dir)
    guesser = ESGuesserWrapper(guesser)
    machine_agent = GuesserBuzzerAgent(guesser, buzzer)

    # setup human agent
    human_agent = HumanAgent()

    # setup hook
    hooks = []
    hooks.append(hook.NotifyBuzzingHook)
    hooks.append(hook.GameInterfaceHook)
    hooks.append(hook.VisualizeGuesserBuzzerHook(machine_agent))
    hooks.append(hook.HighlightHook)

    # setup game
    game = Game(dev_questions, [human_agent, machine_agent], hooks)

    game.run(10)

if __name__ == '__main__':
    main()
