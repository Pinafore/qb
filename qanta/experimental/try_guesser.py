from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
# from qanta.buzzer.rnn_0 import dense_vector

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)

guesses = guesser.guess_single('Sviatoslav Teofilovich Richter was a Soviet pianist \
    known for the depth of his interpretations, virtuoso technique, and vast \
    repertoire.')

print(guesses)

# guesses = [[guesses]]
# vecs = dense_vector(guesses)
# print(vecs)
