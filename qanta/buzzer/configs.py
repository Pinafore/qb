class mlp():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 25
    n_layers = 0
    dropout = 0
    step_size = 1
    neg_weight = 1
    batch_norm = False
    model_name = 'mlp_{}'.format(n_hidden)
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)

class rnn():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 200
    dropout = 0
    step_size = 1
    neg_weight = 1
    model_name = 'rnn_{}'.format(n_hidden)
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)
