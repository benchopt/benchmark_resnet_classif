from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "ResNet classification fitting"

    # XXX: this might be a good spot to specify the size of the ResNet
    parameters = {
    }


    def __init__(self,):
        pass

    def set_data(self, X, y):
        self.X = X
        self.y = y

    def compute(self, beta):
        # TODO: change
        diff = self.y - self.X.dot(beta)
        return .5 * diff.dot(diff)

    def to_dict(self):
        return dict(X=self.X, y=self.y)
