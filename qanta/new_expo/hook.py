class Hook:

    def __init__(self, call_every='round'):
        self.call_every = call_every

class GameInterfaceHook(Hook):
    
    def __init__(self, game, call_every='step'):
        self.call_every = call_every
        self.game = game
        
    def run(self):
        print(self.game.round.get_clue())
        print('++++++++++++')
        print('Scores: ' + ' '.join(str(x) for x in self.game.scores))

class EndOfRoundHook(Hook):

    def __init__(self, game, call_every='round'):
        self.call_every = call_every
        self.game = game

    def run(self):
        print('===== End of Round {} ====='.format(self.game.round_num))
        print('Answer: {}'.format(self.game.round.get_answer()))
        print('Scores: ' + ' '.join(str(x) for x in self.game.scores))
        print()
        print()
        print()
        print()

class NotifyBuzzingHook(Hook):

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


