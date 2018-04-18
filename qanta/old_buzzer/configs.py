class mlp():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 25
    n_layers = 0
    dropout = 0
    step_size = 1
    neg_weight = 1
    batch_norm = False
    make_vector = 'dense_vector_0'
    model_name = 'rnn.{}.{}'.format(n_hidden, make_vector)
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)

class rnn_200_0():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 200
    dropout = 0
    step_size = 1
    neg_weight = 1
    make_vector = 'dense_vector_0'
    model_name = 'rnn.{}.{}'.format(n_hidden, make_vector)
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)

class new_rnn_200():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 200
    dropout = 0
    step_size = 1
    neg_weight = 1
    make_vector = 'dense_vector_0'
    model_name = 'new_rnn_200'
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)

class new_rnn_300():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 300
    dropout = 0
    step_size = 1
    neg_weight = 1
    make_vector = 'dense_vector_0'
    model_name = 'new_rnn_300'
    ckp_dir = 'output/buzzer/{}.ckp'.format(model_name)
    model_dir = 'output/buzzer/{}.npz'.format(model_name)

class rnn_200_1():
    max_grad_norm = 5
    batch_size = 128
    n_hidden = 200
    dropout = 0
    step_size = 1
    neg_weight = 1
    make_vector = 'dense_vector_1'
    model_name = 'rnn.{}.{}'.format(n_hidden, make_vector)
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
