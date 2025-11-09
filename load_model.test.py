from model import Model
from torch import load, tensor, no_grad
from data_sets import species_label

iris_model = Model()

iris_model.load_state_dict(load("iris-ann.pt"))

iris_model.eval()

new_iris1 = tensor([5.9, 3.0, 5.1, 1.8])
new_iris2 = tensor([4.7, 3.2, 1.3, 0.2])

with no_grad():
    print(species_label[iris_model(new_iris1).argmax().item()])
    print(species_label[iris_model(new_iris2).argmax().item()])
