import os
from argparse import Namespace
from qanta.buzzer.util import output_dir

args = Namespace()

args.n_input = 40
args.n_layers = 1
args.n_hidden = 50
args.n_output = 2
args.dropout = 0.4

args.batch_size = 64
args.epoch = 20
args.gpu = 0

args.model_name = 'best_model.npz'
args.model_path = os.path.join(output_dir, args.model_name)
