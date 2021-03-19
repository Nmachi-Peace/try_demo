import re
from numpy import array
import matplotlib.pyplot as plt
import nltk
import neattext.functions as nfx
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, LSTM
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#data = pd.read_csv('emails.csv')


with open('C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\stylo_project.txt', 'r') as f:
    data = f.read()

#print(data)
#print(type(data))

#dataset = list(data.split("  "))
#print(dataset)

dataset = data.strip(' ').split(', ')
#print(dataset)


tokenised = Tokenizer(num_words=1000, oov_token='OOV')
tokenised.fit_on_texts(dataset)
word_index = tokenised.word_index
#print(word_index)



data_Sequences = tokenised.texts_to_sequences(dataset)
#print(train_Sequences)


maxLength = max([len(email) for email in data_Sequences])
#print(maxLength)

vocab_size = 10000
#encoded_emails = [one_hot(d, vocab_size) for d in dataset]
#print(encoded_emails)
# pad documents to a max length of 4 words


padded_data = pad_sequences(data_Sequences, maxlen=maxLength, padding='post')
#print(padded_train)

df = pd.DataFrame(padded_data)
#print(df)

dropRow = df.drop([4])
#print(dropRow)


X = np.array(dropRow)
#print(X)

labels = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
label = to_categorical(labels)
#print(labels)

print(X.shape)
print(label.shape)

train = train_test_split(X,   train_size=0.8,  random_state=4, shuffle=True)
print(len(train))
test = train_test_split(X,  test_size=0.2,  random_state=4, shuffle=True)
print(len(test))


#tdf pd.DataFrame.from_records()

# define the model
dim = 10
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=maxLength))
model.add(Flatten())
model.add(Dense(4, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize the model
#print(model.summary())
# fit the model
model.fit(train, test, label, epochs=20, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(train, test, label, verbose=2)
print('Accuracy: %f' % (accuracy*100))


