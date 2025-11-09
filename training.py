from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from model import model
from torch import tensor, no_grad, save
from matplotlib.pyplot import plot, xlabel, ylabel, title, savefig
from data_sets import X_train, X_test, y_test, y_train, species_label


loss_fn = CrossEntropyLoss()

learning_rate = 0.01
optimizer = Adam(params=model.parameters(), lr=learning_rate)

epoches = 500
losses = []

# for param in model.parameters():
#     param.grad = None

model.train()
for i in range(epoches):
    # prediction
    y_pred = model.forward(X_train)

    # Measure the loss
    loss = loss_fn(y_pred, y_train)  # predicted values vs y_train values

    # keep track of losses
    losses.append(loss.item())

    # back propogation with gredient dicent

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"epoch: {i}, loss: {loss:.10f}")


plot(range(epoches), losses)

xlabel("epoch")
ylabel("loss")
title("loss visuals")
savefig("loss_graph.png")


print()
correct = 0
with no_grad():
    model.eval()
    for i in range(len(X_test)):
        sample = X_test[i]
        y_actual = y_test[i]

        y_pred_test = model.forward(sample)
        predicted = y_pred_test.argmax().item()
        loss = loss_fn(y_pred_test, y_actual)

        if predicted == y_actual:
            correct += 1

        print(
            f"sample: {sample.numpy()}, y_actual: {y_actual.item()}, predicted: {predicted}, loss: {loss:10f}"
        )

    print(
        f"{correct} predictions are correct out of {len(y_test)}, accuracy is {(correct/len(y_test) * 100):.2f}%"
    )


# test
print()
new_iris1 = tensor([5.9, 3.0, 5.1, 1.8])
new_iris2 = tensor([4.7, 3.2, 1.3, 0.2])
with no_grad():
    model.eval()
    print(species_label[model(new_iris1).argmax().item()])
    print(species_label[model(new_iris2).argmax().item()])


# save model
save(model.state_dict(), "iris-ann.pt")
