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

guessers "Dan" {
  class = "qanta.guesser.dan_tf.DANGuesser"
  luigi_dependency = "qanta.pipeline.guesser.dan.DANDependencies"
  enabled = true
  min_appearances = 2
  expand_glove = true
  n_hidden_layers = 1
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
  rnn_cell = "lstm"
}

guessers "KerasDAN" {
  class = "qanta.guesser.dan.DANGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  min_answers = 2
  expand_we = true
  n_hidden_layers = 1
  n_hidden_units = 300
  dropout_probability = 0.5
}
