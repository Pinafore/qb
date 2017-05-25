import os
import gzip


class Source:
    def __init__(self):
        self.source_path = 'data/internal/source'

    def __getitem__(self, item):
        filename = os.path.join(self.source_path, item.replace('_', ''))
        if os.path.exists(filename):
            with gzip.open(filename) as f:
                text = f.read().decode('utf8')
            return text
        else:
            return ''
