from argparse import Namespace

args = Namespace()

args.n_input = 12718
args.n_layers = 1
args.n_hidden = 50
args.n_output = 2
args.dropout = 0.4

args.batch_size = 64
args.epoch = 20
args.gpu = 0

args.outdir = 'output/buzzer'
