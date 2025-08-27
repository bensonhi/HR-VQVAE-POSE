class CycleScheduler:
    def __init__(self, optimizer, lr, n_iter, momentum=None):
        self.optimizer = optimizer
        self.lr = lr
        self.n_iter = n_iter
        self.current_iter = 0
        
    def step(self):
        self.current_iter += 1
        # Simple constant learning rate for now
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr