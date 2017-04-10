n_guesses = 50

test_fold_word_skip = -1

generate_train_guesses = false

guessers_train_on_dev = false

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
}

guessers "Whoosh" {
  class = "qanta.guesser.experimental.whoosh.WhooshWikiGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
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
  n_cores = 2
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
}
