from pandas import read_csv
from sklearn.model_selection import train_test_split
from model import random_seed
from torch import tensor, float32

df = read_csv("iris.csv")

if __name__ == "__main__":
    print(df.columns)

species_label = df["species"].unique().tolist()

if __name__ == "__main__":
    print("species_label", species_label)

# df["species"].replace(species_label[1], 1.0, inplace=True)
# df["species"].replace(species_label[2], 2.0, inplace=True)


df.replace(
    {
        "species": {
            species_label[0]: 0.0,
            species_label[1]: 1.0,
            species_label[2]: 2.0,
        }
    },
    inplace=True,
)


# print(df.head().to_numpy())
# print(df.tail().to_numpy())

X = df.drop("species", axis=1).to_numpy()
y = df["species"].to_numpy()

# print(X.head())
# print(y.tail())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, test_size=0.2, random_state=random_seed
)

# X_train = tensor(X_train, dtype=float16)
# X_test = tensor(X_test, dtype=float16)
X_train = tensor(X_train, dtype=float32)
X_test = tensor(X_test, dtype=float32)
# X_train = tensor(X_train, dtype=float64)
# X_test = tensor(X_test, dtype=float64)
y_train = tensor(y_train, dtype=int)
y_test = tensor(y_test, dtype=int)


if __name__ == "__main__":
    print(X_train.shape, y_train.shape)
