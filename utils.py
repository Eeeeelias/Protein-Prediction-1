# custom early stopping, based on chosen metric, works for minimizing metrics
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = current_score
            self.counter = 0