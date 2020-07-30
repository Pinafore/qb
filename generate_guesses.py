import pickle
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.dan import DanGuesser
from qanta.util.constants import BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD

guesser_directory = AbstractGuesser.output_path(
    "qanta.guesser.dan", "DanGuesser", 0, ""
)
guesser = DanGuesser.load(guesser_directory)  # type: AbstractGuesser
guesser.batch_size /= 8

word_skip = 2
folds = [BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD]
for fold in folds:
    df = guesser.generate_guesses(1, [fold], word_skip=word_skip)
    output_path = AbstractGuesser.guess_path(guesser_directory, fold)
    with open(output_path, "wb") as f:
        pickle.dump(df, f)
