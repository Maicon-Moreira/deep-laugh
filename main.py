import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Reshape, Activation, Dropout
import numpy as np
from random import choice
from keras.utils.vis_utils import plot_model

np.set_printoptions(suppress=True)

laughs = []
with open('sample_laughs.txt', 'r') as file:
    laughs = file.read().split('\n')

num_laughs = len(laughs)

letters = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26,
           'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, ' ': 52, '': 53}
inverted_letters = {v: k for k, v in letters.items()}

print(inverted_letters)


def word_to_array(word):
    array = []
    for letter in word:
        array.append(letters[letter])
    while len(array) < 50:
        array.append(53)
    return array


def array_to_word(array):
    word = ''
    for number in array:
        word += inverted_letters[number]

    return word


model = Sequential([
    # Embedding(54, 10, input_length=50),
    # LSTM(128)
    # Dense(10, input_shape=(3,))

    Dense(100, input_shape=(num_laughs,)),
    # Activation('relu'),
    # Dropout(0.1),
    Dense(500),
    # Activation('sigmoid'),
    # Dropout(0.3),
    Dense(50*54),
    Reshape((50, 54))
])
model.compile('adam', 'mse')
model.summary()
# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model.png')


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

print(xs.shape)
print(ys.shape)

model.fit(xs, ys, epochs=10)


created_laughs = []
for i in range(100):
    bias = np.random.rand(1, num_laughs)**3
    # print(bias)

    array = model.predict(bias)[0]
    array = [np.argmax(item) for item in array]
    word = array_to_word(array)
    created_laughs.append(word)

with open('created_laughs.txt', 'w') as file:
    text = ''
    for laugh in created_laughs:
        text += laugh+'\n'
    file.write(text)