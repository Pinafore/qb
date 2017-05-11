class mlp():
    optimizer = 'Adam'
    lr = 1e-3
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 200
    n_layers = 3
    dropout = 0.3
    batch_norm = True
    log_dir = 'mlp_buzzer.log'
    ckp_dir = 'mlp_buzzer.ckp'
    model_dir = 'output/buzzer/mlp_buzzer.npz'

class rnn():
    optimizer = 'Adam'
    lr = 1e-3
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 128
    log_dir = 'rnn_buzzer.log'
    ckp_dir = 'rnn_buzzer.ckp'
    model_dir = 'output/buzzer/rnn_buzzer.npz'
