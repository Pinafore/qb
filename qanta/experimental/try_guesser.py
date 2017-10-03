from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
from qanta.buzzer.rnn_0 import dense_vector0 as dense_vector

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)

guesses = guesser.guess_single("wrote a novel about a boy who encounters the\
        Duke and the Dauphin while traveling down the Mississippi with the slave\
        Jim")

guesses = [[guesses]]
print(guesses)
vecs = dense_vector(guesses)
print(vecs)

