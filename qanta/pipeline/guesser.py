import pickle
import os
import time

import luigi
from luigi import LocalTarget, Task, WrapperTask

from qanta.config import conf
from qanta.util import constants as c
from qanta.util.io import shell
from qanta.guesser.abstract import AbstractGuesser, get_class
from qanta.pipeline.preprocess import DownloadData
from qanta import qlogging

log = qlogging.get(__name__)


class EmptyTask(luigi.Task):
    def complete(self):
        return True


class TrainGuesser(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    config_num = luigi.IntParameter()  # type: int

    def requires(self):
        yield DownloadData()
        if self.dependency_class is not None and self.dependency_module is not None:
            dependency_class = get_class(self.dependency_module, self.dependency_class)
            yield dependency_class()

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_instance = guesser_class(self.config_num)  # type: AbstractGuesser
        qb_dataset = guesser_instance.qb_dataset()
        start_time = time.time()
        guesser_instance.train(qb_dataset.training_data())
        end_time = time.time()
        guesser_instance.save(AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, ''
        ))
        params = guesser_instance.parameters()
        params['training_time'] = end_time - start_time
        params_path = AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, 'guesser_params.pickle'
        )
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)

    def output(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_targets = [
            LocalTarget(file)
            for file in guesser_class.files(
                AbstractGuesser.output_path(self.guesser_module, self.guesser_class, self.config_num, '')
            )]

        return [
            LocalTarget(AbstractGuesser.output_path(
                self.guesser_module, self.guesser_class, self.config_num, ''
            )),
            LocalTarget(
                AbstractGuesser.output_path(
                    self.guesser_module, self.guesser_class, self.config_num, 'guesser_params.pickle'
                ))
        ] + guesser_targets


class TrainAllGuessers(WrapperTask):

    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            yield TrainGuesser(
                guesser_module=g_spec.guesser_module,
                guesser_class=g_spec.guesser_class,
                dependency_module=g_spec.dependency_module,
                dependency_class=g_spec.dependency_class
            )


class GenerateGuesses(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    config_num = luigi.IntParameter()  # type: int
    n_guesses = luigi.IntParameter(default=conf['n_guesses'])  # type: int
    fold = luigi.Parameter()  # type: str

    def requires(self):
        yield TrainGuesser(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class,
            config_num=self.config_num
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, ''
        )
        guesser_instance = guesser_class.load(guesser_directory)  # type: AbstractGuesser

        if self.fold in {c.GUESSER_TRAIN_FOLD, c.GUESSER_DEV_FOLD}:
            char_skip = conf['guesser_char_skip']
        else:
            char_skip = conf['buzzer_char_skip']

        log.info(f'Generating and saving guesses for {self.fold} fold with char_skip={char_skip}...')
        start_time = time.time()
        guess_df = guesser_instance.generate_guesses(self.n_guesses, [self.fold], char_skip=char_skip)
        end_time = time.time()
        elapsed = end_time - start_time
        log.info(f'Guessing on {self.fold} fold took {elapsed}s, saving guesses...')
        guesser_class.save_guesses(guess_df, guesser_directory, [self.fold], 'char')
        log.info('Done saving guesses')

        log.info(f'Generating and saving guesses for {self.fold} fold with full question...')
        start_time = time.time()
        guess_df = guesser_instance.generate_guesses(self.n_guesses, [self.fold], full_question=True)
        end_time = time.time()
        elapsed = end_time - start_time
        log.info(f'Guessing on {self.fold} fold took {elapsed}s, saving guesses...')
        guesser_class.save_guesses(guess_df, guesser_directory, [self.fold], 'full')
        log.info('Done saving guesses')

        log.info(f'Generating and saving guesses for {self.fold} fold with first sentence')
        start_time = time.time()
        guess_df = guesser_instance.generate_guesses(self.n_guesses, [self.fold], first_sentence=True)
        end_time = time.time()
        elapsed = end_time - start_time
        log.info(f'Guessing on {self.fold} fold took {elapsed}s, saving guesses...')
        guesser_class.save_guesses(guess_df, guesser_directory, [self.fold], 'first')
        log.info('Done saving guesses')

    def output(self):
        files = [
            f'guesses_char_{self.fold}.pickle',
            f'guesses_full_{self.fold}.pickle',
            f'guesses_first_{self.fold}.pickle'
        ]
        return [
            LocalTarget(AbstractGuesser.output_path(
                self.guesser_module, self.guesser_class, self.config_num, f
        )) for f in files]


class GenerateAllGuesses(WrapperTask):
    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            for fold in c.GUESSER_GENERATION_FOLDS:
                yield GenerateGuesses(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    config_num=g_spec.config_num,
                    fold=fold
                )


class GuesserReport(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    config_num = luigi.IntParameter()  # type: int
    fold = luigi.Parameter()  # type: str

    def requires(self):
        yield GenerateGuesses(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class,
            config_num=self.config_num,
            fold=self.fold
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        guesser_directory = AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, ''
        )
        guesser_instance = guesser_class(self.config_num)
        guesser_instance.create_report(guesser_directory, self.fold)

    def output(self):
        return LocalTarget(AbstractGuesser.output_path(
            self.guesser_module,
            self.guesser_class,
            self.config_num,
            f'guesser_report_{self.fold}.pickle'
        ))


class GuesserPerformance(Task):
    guesser_module = luigi.Parameter()  # type: str
    guesser_class = luigi.Parameter()  # type: str
    dependency_module = luigi.Parameter()  # type: str
    dependency_class = luigi.Parameter()  # type: str
    config_num = luigi.IntParameter()  # type: int
    fold = luigi.Parameter()  # type: str

    def requires(self):
        yield GenerateGuesses(
            guesser_module=self.guesser_module,
            guesser_class=self.guesser_class,
            dependency_module=self.dependency_module,
            dependency_class=self.dependency_class,
            config_num=self.config_num,
            fold=self.fold
        )

    def run(self):
        guesser_class = get_class(self.guesser_module, self.guesser_class)
        reporting_directory = AbstractGuesser.reporting_path(
            self.guesser_module, self.guesser_class, self.config_num, ''
        )

        # In the cases of huge parameter sweeps on SLURM its easy to accidentally run out of /fs/ storage.
        # Since we only care about the results we can get them, then delete the models. We can use the regular
        # GuesserReport to preserve the model
        guesser_directory = AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, ''
        )

        param_path = AbstractGuesser.output_path(
            self.guesser_module, self.guesser_class, self.config_num, f'guesser_params.pickle'
        )
        guesses_files = [
            f'guesses_char_{self.fold}.pickle',
            f'guesses_full_{self.fold}.pickle',
            f'guesses_first_{self.fold}.pickle'
        ]
        guesses_paths = [
            AbstractGuesser.output_path(
                self.guesser_module, self.guesser_class, self.config_num, f)
            for f in guesses_files
        ]

        log.info(f'Running: "cp {param_path} {reporting_directory}"')
        shell(f'cp {param_path} {reporting_directory}')

        for g_path in guesses_paths:
            log.info(f'Running: "cp {g_path} {reporting_directory}"')
            shell(f'cp {g_path} {reporting_directory}')

        guesser_instance = guesser_class(self.config_num)
        guesser_instance.create_report(reporting_directory, self.fold)

        log.info(f'Running: "rm -rf {guesser_directory}"')
        shell(f'rm -rf {guesser_directory}')
        for g_path in guesses_paths:
            shell(f'rm -f {g_path}')

    def output(self):
        return [
            LocalTarget(AbstractGuesser.reporting_path(
                self.guesser_module,
                self.guesser_class,
                self.config_num,
                f'guesser_report_{self.fold}.pickle'
            )),
            LocalTarget(AbstractGuesser.reporting_path(
                self.guesser_module,
                self.guesser_class,
                self.config_num,
                f'guesser_params.pickle'
            ))
        ]


class AllGuesserPerformance(WrapperTask):
    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            for fold in [c.GUESSER_DEV_FOLD, c.GUESSER_TEST_FOLD]:
                yield GuesserPerformance(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    config_num=g_spec.config_num,
                    fold=fold
                )
            if os.path.exists(c.QANTA_EXPO_DATASET_PATH):
                yield GuesserPerformance(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    config_num=g_spec.config_num,
                    fold=c.EXPO_FOLD
                )


class AllGuesserReports(WrapperTask):
    def requires(self):
        for g_spec in AbstractGuesser.list_enabled_guessers():
            for fold in [c.GUESSER_DEV_FOLD, c.GUESSER_TEST_FOLD]:
                yield GuesserReport(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    config_num=g_spec.config_num,
                    fold=fold
                )
            if os.path.exists(c.QANTA_EXPO_DATASET_PATH):
                yield GuesserReport(
                    guesser_module=g_spec.guesser_module,
                    guesser_class=g_spec.guesser_class,
                    dependency_module=g_spec.dependency_module,
                    dependency_class=g_spec.dependency_class,
                    config_num=g_spec.config_num,
                    fold=c.EXPO_FOLD
                )
