class config:
    def __init__(self):
        self.optimizer = 'Adam'
        self.lr = 1e-3
        self.max_grad_norm = 5
        self.batch_size = 128

class mlp(config):
    def __init__(self):
        super(mlp, self).__init__()
        self.n_hidden = 200
        self.n_layers = 3
        self.dropout = 0.3
        self.log_dir = 'mlp_buzzer.log'
        self.model_dir = 'output/buzzer/mlp_buzzer.npz'

class rnn(config):
    def __init__(self):
        super(rnn, self).__init__()
        self.n_hidden = 128
        self.log_dir = 'rnn_buzzer.log'
        self.model_dir = 'output/buzzer/rnn_buzzer.npz'
