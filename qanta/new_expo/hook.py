from display_util import show_score
from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections

class Hook:
    '''A Hook object is called by the game control at the end of each step or
    each round, which is indicated by call_every'''

    def __init__(self, call_every='round'):
        self.call_every = call_every

class GameInterfaceHook(Hook):
    '''The old digit interface'''
    
    def __init__(self, game, call_every='step'):
        self.call_every = call_every
        self.game = game
        
    def run(self):
        print(self.game.round.question.page)

        print('++++++++++++')
        print(self.game.round.get_answer())
        print('------')
        print(self.game.round.get_clue())
        show_score(self.game.scores[0], self.game.scores[1], flush=False)

class NotifyBuzzingHook(Hook):
    '''Notifies all agents of the buzzing state using each agent's
    notify_buzzing function'''

    def __init__(self, game, call_every='step'):
        self.call_every = call_every
        self.game = game

    def run(self):
        buzzed = self.game.buzzed
        agents = self.game.agents
        for i, agent in enumerate(agents):
            # move each agent to the first
            b = [buzzed[i]] + buzzed[:i] + buzzed[i+1:]
            agent.notify_buzzing(buzzed)

class VisualizeGuesserBuzzerHook(Hook):
    '''Show the list of guesses and the buzzer score'''

    def __init__(self, guesser_buzzer, call_every='step'):
        self.guesser = guesser_buzzer.guesser
        self.buzzer = guesser_buzzer.buzzer
        self.call_every = call_every

    def run(self):
        print('===== Guesser =====')
        for guess, score in self.guesser.guesses:
            print(guess, score)
        print('===== Buzzer =====')
        print(self.buzzer.ys)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class HighlightHook(Hook):

    def __init__(self, game, call_every='step'):
        connections.create_connection(hosts=['localhost'])
        self.game = game
        self.call_every = call_every

    def get_highlights(self, text):
        # query top 10 guesses
        s = Search(index='qb_ir_instance_of')[0:10].query('multi_match', query=text,
                fields=['wiki_content', 'qb_content', 'source_content'])
        s = s.highlight('qb_content').highlight('wiki_content')
        results = list(s.execute())
        guess = results[0] # take the best answer
        _highlights = guess.meta.highlight 
    
        try:
            wiki_content = list(_highlights.wiki_content)
        except AttributeError:
            wiki_content = None
    
        try:
            qb_content = list(_highlights.qb_content)
        except AttributeError:
            qb_content = None

        highlights = {'wiki': wiki_content,
                      'qb': qb_content,
                      'guess': guess.page}
        return highlights
    
    def run(self):
        question = self.game.round.get_clue()
        highlights = self.get_highlights(question)
        if highlights['wiki']:
            for x in highlights['wiki']:
                print('WIKI|' + x.replace('<em>', color.RED).replace('</em>', color.END))
        if highlights['qb']:
            for x in highlights['qb']:
                print('QUIZ|' + x.replace('<em>', color.BLUE).replace('</em>', color.END))

