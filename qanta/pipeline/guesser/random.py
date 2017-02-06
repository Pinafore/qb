import luigi


class EmptyTask(luigi.Task):
    def complete(self):
        return True
