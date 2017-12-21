n_guesses = 50
guesser_word_skip = -1
buzzer_word_skip = 2

expo_mega_hack = true # yes really
expo_questions = "data/external/2017_hsnct.csv"

word_embeddings = "data/external/deep/glove.6B.300d.txt"
embedding_dimension = 300
use_pretrained_embeddings = true
buzz_as_guesser_train = false

# Configure whether qanta.wikipedia.cached_wikipedia.CachedWikipedia should fallback
# performing a remote call to Wikipedia if a page doesn't exist
cached_wikipedia_remote_fallback = false


guessers "ElasticSearch" {
  class = "qanta.guesser.elasticsearch.ElasticSearchGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
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

guessers "Dan" {
  class = "qanta.guesser.dan.DanGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false

  use_wiki = false
  use_tagme = false
  n_tagme_sentences = 20

  optimizer = "adam"
  batch_size = 1024
  max_epochs = 100
  gradient_clip = 0.25
  sgd_weight_decay = 0.0000012
  sgd_lr = 30.0
  adam_lr = 0.001
  use_lr_scheduler = true
  nn_dropout = 0.265
  sm_dropout = 0.158
  hyper_opt = false
  dual_encoder = false
}


guessers "EntityRNN" {
  class = "qanta.guesser.rnn_entity.RnnEntityGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = false
  features = ["word"]

  max_epochs = 100
  batch_size = 256
  learning_rate = 0.001
  max_grad_norm = 5
  rnn_type = "gru"
  dropout_prob = 0.25
  variational_dropout_prob = 0.25
  bidirectional = true
  n_hidden_units = 1000
  n_hidden_layers = 1
  use_wiki = false
  use_triviaqa = false
  use_tagme = false
  n_tagme_sentences = 20
  n_wiki_paragraphs = 3
  sm_dropout_prob = 0.15
  sm_dropout_before_linear = false
  use_cove = false
}

guessers "Bcn" {
    class = "qanta.guesser.bcn.BcnGuesser"
    luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
    enabled = false
}

guessers "ESWikidata" {
  class = "qanta.guesser.experimental.elasticsearch_instance_of.ElasticSearchWikidataGuesser"
  luigi_dependency = "qanta.pipeline.guesser.EmptyTask"
  enabled = true
  # Set the level of parallelism for guess generation
  n_cores = 20
  confidence_threshold = 0.7
  normalize_score_by_length = true
}

guessers "VowpalWabbit" {
  class = "qanta.guesser.vw.VWGuesser"
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
  config="rnn"
}
