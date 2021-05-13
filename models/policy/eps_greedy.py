from models.policy import DecayingEpsilonGreedy


class EpsilonGreedy(DecayingEpsilonGreedy):
    def __init__(self,
                 eps):
        super(EpsilonGreedy, self).__init__(
            eps_decay=1,
            eps_start=eps
        )

    def update(self):
        pass
