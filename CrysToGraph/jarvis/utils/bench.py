from .jarvis_tasks import DATASETS_TASKS


class MatbenchBenchmark:
    def __init__(self, **kwargs):
        for k in DATASETS_TASKS.keys():
            setattr(self, k, DATASETS_TASKS[k])

    @property
    def tasks(self):
        return DATASETS_TASKS.values()

    def from_preset(self, name, **kwargs):
        return self

    def add_metadata(self, metadata, **kwargs):
        return 0

    def to_file(self, path, **kwargs):
        return 0
