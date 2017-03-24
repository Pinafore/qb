n_guesses = 50

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
  enabled = false
  # Set the level of parallelism for guess generation
  n_cores = 15
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
