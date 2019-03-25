class LabelSet(object):
    def __init__(self, name, labels, payload_name, task_name, source_name=None):
        self.name = name
        self.labels = labels
        self.payload_name = payload_name
        self.task_name = task_name
        self.source_name = source_name
