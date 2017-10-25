n_guesses = 50
guesser_word_skip = -1
buzzer_word_skip = 2

expo_questions = "data/internal/expo/2016_hsnct.csv"

word_embeddings = "data/external/deep/glove.6B.300d.txt"
embedding_dimension = 300
use_pretrained_embeddings = true

# Configure whether qanta.wikipedia.cached_wikipedia.CachedWikipedia should fallback
# performing a remote call to Wikipedia if a page doesn't exist
cached_wikipedia_remote_fallback = false

wiki_data_frac = 0.0

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
  use_source = false
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

guessers "DAN" {
  class = "qanta.guesser.dan.DANGuesser"
  luigi_dependency = "qanta.pipeline.wiki_questions.SelectWikiQuestions"
  enabled = false
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
  max_len = 200
  output_last_hidden = false
}

guessers "PTDan" {
  class = "qanta.guesser.torch.dan.DanGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "PTRnn" {
  class = "qanta.guesser.torch.rnn.RnnGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "EntityRNN" {
  class = "qanta.guesser.torch.rnn_entity.RnnEntityGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
}

guessers "ESWikidata" {
  class = "qanta.guesser.experimental.elasticsearch_instance_of.ElasticSearchWikidataGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  # Set the level of parallelism for guess generation
  n_cores = 20
  confidence_threshold = 0.7
  normalize_score_by_length = true
}

guessers "CNN" {
  class = "qanta.guesser.cnn.CNNGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  expand_we = true
  n_filter_list = [10]
  filter_sizes = [2, 3, 4]
  nn_dropout_rate = 0.5
  batch_size = 512
  learning_rate = 0.001
  max_n_epochs = 100
  max_patience = 10
  activation_function = "relu"
  train_on_q_runs = false
  train_on_full_q = false
  decay_lr_on_plateau = false
  max_len = 200
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
