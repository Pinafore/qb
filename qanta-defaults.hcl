n_guesses = 50

test_fold_word_skip = -1

generate_train_guesses = false

guessers_train_on_dev = false

word_embeddings = "data/external/deep/glove.6B.300d.txt"
embedding_dimension = 300
use_pretrained_embeddings = true

clm {
  min_appearances = 2
}

wikifier {
  min_appearances = 2
}

guessers "AuxDan" {
  class = "qanta.guesser.experimental.dan.aux_dan.AuxDANGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "ElasticSearch" {
  class = "qanta.guesser.elasticsearch.ElasticSearchGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = true
  # Set the level of parallelism for guess generation
  n_cores = 15
  min_appearances = 1
}

guessers "ElasticSearchWikidata" {
  class = "qanta.guesser.elasticsearch_wikidata.ElasticSearchWikidataGuesser"
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

guessers "RNN" {
  class = "qanta.guesser.rnn.RNNGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  min_answers = 2
  expand_we = true
  rnn_cell = "gru"
  n_rnn_units = 300
  max_patience = 10
  max_n_epochs = 100
  batch_size = 512
}

guessers "KerasDAN" {
  class = "qanta.guesser.dan.DANGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = true
  min_answers = 2
  expand_we = true
  n_hidden_layers = 1
  n_hidden_units = 300
  nn_dropout_rate = 0.5
  word_dropout_rate = 0.5
  batch_size = 256
  learning_rate = 0.001
  l2_normalize_averaged_words = false
  max_n_epochs = 100
  max_patience = 10
  activation_function = "relu"
}
