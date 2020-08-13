class History(object):
    def __init__(self, anno):
        self.anno = anno
        self.correctness = 0
        self.confidence = 0
        self.max_correctness = 1

    def max_correctness_update(self):
        self.max_correctness += 1


