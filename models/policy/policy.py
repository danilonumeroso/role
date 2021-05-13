class BasePolicy:

    def take(self, q_approximator, states):
        raise NotImplementedError

    def update(self):
        pass
