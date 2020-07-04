import keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout
import numpy as np
import os

# Init wandb
import wandb
from wandb.keras import WandbCallback
wandb.init(project="deep-laugh")

np.set_printoptions(suppress=True)

laughs = []
with open('sample_laughs.txt', 'r') as file:
    laughs = file.read().split('\n')

num_laughs = len(laughs)

letters = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26,
           'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, ' ': 52, '': 53}
inverted_letters = {v: k for k, v in letters.items()}


def word_to_array(word):
    array = []
    for letter in word:
        array.append(letters[letter])
    while len(array) < 50:
        array.append(53)
    return array


model = Sequential([
    Dense(100, input_shape=(num_laughs,)),
    Dense(500),
    Dense(50*54),
    Reshape((50, 54))
])
model.compile('adam', 'mse')
model.summary()

xs = []
ys = []

for i in range(1000):
    laugh_i = np.random.randint(num_laughs)

    xs.append(keras.utils.to_categorical(laugh_i, num_laughs)[0])

    laugh = laughs[laugh_i]
    laugh = word_to_array(laugh)
    laugh = [keras.utils.to_categorical(item, 54)[0] for item in laugh]

    ys.append(laugh)

xs = np.array(xs)
ys = np.array(ys)

model.fit(xs, ys, epochs=10, callbacks=[WandbCallback()])

model.save('model.h5')
model.save(os.path.join(wandb.run.dir, "model.h5"))
