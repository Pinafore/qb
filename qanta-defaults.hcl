n_guesses = 50
guesser_word_skip = -1
buzzer_word_skip = 2

expo_questions = "data/internal/expo/2016_hsnct.csv"

word_embeddings = "data/external/deep/glove.6B.300d.txt"
embedding_dimension = 300
use_pretrained_embeddings = true
buzz_as_guesser_train = false


guessers "ElasticSearch" {
  class = "qanta.guesser.elasticsearch.ElasticSearchGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = true
  # Set the level of parallelism for guess generation
  n_cores = 15
  # Whether or not to index all Wikipedia articles for guessing
  use_all_wikipedia = false
  use_wiki = true
  use_qb = true
  many_docs = false
  normalize_score_by_length = true
  wiki_boost = 1
  qb_boost = 1
}

guessers "Tfidf" {
  class = "qanta.guesser.tfidf.TfidfGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "Dan" {
  class = "qanta.guesser.dan.DanGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false

  use_wiki = false
  use_tagme = false
  n_tagme_sentences = 20
  n_wiki_sentences = 5
  qb_fraction = 1.0

  n_hidden_units = 1000
  n_hidden_layers = 1

  optimizer = "adam"
  batch_size = 1024
  max_epochs = 100
  gradient_clip = 0.25
  sgd_weight_decay = 0.0000012
  sgd_lr = 30.0
  adam_lr = 0.001
  adam_weight_decay = 0.0
  use_lr_scheduler = true
  nn_dropout = 0.265
  sm_dropout = 0.158
  hyper_opt = false
  hyper_opt_steps = 300
  dual_encoder = false

  wiki_training = "mixed" # Options: "mixed", "pretrain"
}

guessers "Tied" {
  class = "qanta.guesser.tied.TiedGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false

  gradient_clip = 0.25
  n_hidden_units = 1000
  n_hidden_layers = 1
  lr = 0.001
  nn_dropout = 0.265
  sm_dropout = 0.158
  batch_size = 1024
  lowercase = true

  use_wiki = false
  n_wiki_sentences = 5
  wiki_title_replace_token = ""
  tied_l2 = 0.0
  bigrams = false
}

guessers "EntityRNN" {
  class = "qanta.guesser.rnn_entity.RnnEntityGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  features = ["word"]

  max_epochs = 100
  batch_size = 256
  learning_rate = 0.001
  weight_decay = 0
  max_grad_norm = 1.0
  rnn_type = "gru"
  dropout_prob = 0.25
  variational_dropout_prob = 0.25
  bidirectional = true
  n_hidden_units = 1000
  n_hidden_layers = 1
  sm_dropout_prob = 0.15
  sm_dropout_before_linear = false

  use_cove = false
  use_locked_dropout = false

  use_wiki = false
  use_triviaqa = false
  use_tagme = false
  n_tagme_sentences = 20
  n_wiki_sentences = 10

  hyper_opt = false
  hyper_opt_steps = 100
}

buzzer {
  n_cores=16
  n_guesses=50
  gpu=0
  config="mlp"
}
