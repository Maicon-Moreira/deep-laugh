import keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout
import numpy as np
import os
from tqdm import tqdm

np.set_printoptions(suppress=True)

laughs = []
with open('sample_laughs.txt', 'r') as file:
    laughs = file.read().split('\n')

num_laughs = len(laughs)

letters = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26,
           'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, ' ': 52, '': 53}
inverted_letters = {v: k for k, v in letters.items()}


def array_to_word(array):
    word = ''
    for number in array:
        word += inverted_letters[number]

    return word


model = keras.models.load_model('model.h5')

created_laughs = []
for i in tqdm(range(1000)):
    bias = np.random.rand(1, num_laughs)**3
    array = model.predict(bias)[0]
    array = [np.argmax(item) for item in array]
    word = array_to_word(array)
    created_laughs.append(word)

with open('created_laughs.txt', 'w') as file:
    text = ''
    for laugh in created_laughs:
        text += laugh+'\n'
    file.write(text)
