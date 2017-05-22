n_guesses = 50
guesser_word_skip = -1
buzzer_word_skip = -1

expo_questions = "data/external/expo.csv"

word_embeddings = "data/external/deep/glove.6B.300d.txt"
embedding_dimension = 300
use_pretrained_embeddings = true

# Configure whether qanta.wikipedia.cached_wikipedia.CachedWikipedia should fallback
# performing a remote call to Wikipedia if a page doesn't exist
cached_wikipedia_remote_fallback = false

wiki_data_frac = 0.0

clm {
  min_appearances = 2
}

wikifier {
  min_appearances = 2
}

guessers "ElasticSearch" {
  class = "qanta.guesser.elasticsearch.ElasticSearchGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = true
  # Set the level of parallelism for guess generation
  n_cores = 15
  min_appearances = 1
  # Whether or not to index all Wikipedia articles for guessing
  use_all_wikipedia = false
  use_wiki = true
  use_qb = true
}

guessers "DAN" {
  class = "qanta.guesser.dan.DANGuesser"
  luigi_dependency = "qanta.pipeline.wiki_questions.SelectWikiQuestions"
  enabled = false
  min_answers = 1
  expand_we = true
  n_hidden_layers = 1
  n_hidden_units = 1000
  nn_dropout_rate = 0.5
  word_dropout_rate = 0.5
  batch_size = 512
  learning_rate = 0.001
  l2_normalize_averaged_words = true
  max_n_epochs = 100
  max_patience = 10
  activation_function = "elu"
  train_on_q_runs = false
  train_on_full_q = false
  decay_lr_on_plateau = false
  generate_mentions = false
}

guessers "RNN" {
  class = "qanta.guesser.rnn.RNNGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  min_answers = 1
  expand_we = true
  rnn_cell = "gru"
  n_rnn_units = 300
  max_patience = 10
  max_n_epochs = 100
  batch_size = 64
  nn_dropout_rate = 0.5
  n_rnn_layers = 1
  bidirectional_rnn = false
  # The default is to train on sentences
  train_on_q_runs = false
  train_on_full_q = false
  decay_lr_on_plateau = false
}

guessers "MemNN" {
  class = "qanta.guesser.mem_nn.MemNNGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  min_answers = 2
  expand_we = true
  n_hops = 1
  n_hidden_units = 300
  nn_dropout_rate = 0.5
  word_dropout_rate = 0.5
  batch_size = 256
  learning_rate = 0.001
  l2_normalize_averaged_words = true
  max_n_epochs = 100
  max_patience = 10
  activation_function = "elu"
  n_wiki_sentences = 10
  n_memories = 10
  n_cores = 2
}

guessers "AuxDan" {
  class = "qanta.guesser.experimental.dan.aux_dan.AuxDANGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "ElasticSearchWikidata" {
  class = "qanta.guesser.experimental.elasticsearch_wikidata.ElasticSearchWikidataGuesser"
  luigi_dependency = "qanta.pipeline.guesser.wikidata.DownloadWikidata"
  enabled = false
  # Set the level of parallelism for guess generation
  n_cores = 15
  min_appearances = 1
}

guessers "FixedLen" {
  class = "qanta.guesser.experimental.tf_fixed.FixedLenGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "CNN" {
  class = "qanta.guesser.experimental.cnn.CNNGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "BinarizedSiamese" {
  class = "qanta.guesser.experimental.binarized.BinarizedGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false

  # Model parameters
  nn_dropout_keep_prob = 0.6
}

guessers "VowpalWabbit" {
  class = "qanta.guesser.experimental.vw.VWGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false

  # These two flags are XOR with each other, one must be true and the other false
  multiclass_one_against_all = false
  multiclass_online_trees = true
  l2 = 0.000001
  l1 = 0
  passes = 20
  learning_rate = 0.1
  decay_learning_rate = 0.95
  bits = 30
}


buzzer {
  n_cores=16
  n_guesses=50
  gpu=0
  config="mlp"
}
