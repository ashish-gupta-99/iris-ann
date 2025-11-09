from torch.nn import Module
from torch.nn import Linear
from torch.nn.functional import relu
from torch import manual_seed, float32
from numpy.random import seed as np_seed
from random import seed as r_seed

random_seed = 10

class Model(Module):
    def __init__(self, in_features=4, h1_layer=6, h2_layer=8, h3_layer=10, out_features=3, dtype=float32):
        super().__init__()
        self.fully_connected1 = Linear(in_features=in_features, out_features=h1_layer, dtype=dtype)
        self.fully_connected2 = Linear(in_features=h1_layer, out_features=h2_layer, dtype=dtype)
        self.fully_connected3 = Linear(in_features=h2_layer, out_features=h3_layer, dtype=dtype)
        self.output = Linear(in_features=h3_layer, out_features=out_features, dtype=dtype)

    def forward(self, x):
        x1 = relu(self.fully_connected1(x))
        x2 = relu(self.fully_connected2(x1))
        x3 = relu(self.fully_connected3(x2))
        x_out = relu(self.output(x3))

        return x_out


# pick a manual seed for randomisation
manual_seed(random_seed)
np_seed(random_seed)
r_seed(random_seed)


model = Model()
