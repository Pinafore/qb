import luigi


@luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
def get_execution_time(self, processing_time):
    self.execution_time = processing_time
