import luigi


class EmptyTask(luigi.ExternalTask):
    def requires(self):
        return []
