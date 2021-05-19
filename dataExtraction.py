import os
import shutil
import glob
import email
import pandas as pd

from numpy import array
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
import email
from nltk.tokenize import word_tokenize
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from os import makedirs
from keras.models import load_model
from numpy import dstack
from sklearn.linear_model import LogisticRegression


#input = glob.glob('C:\\Users\\Nmachi\Desktop\\PycharmProjects\\test\enron_mail_20150507.tar\\enron_mail_20150507\\maildir')
#outpout = 'C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\dest'

#for f in input:
    #shutil.move(f, outpout)

#Define the main data
'''emails = pd.read_csv("C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\emails.csv")
#print(type(emails))

#print(emails.head())

#print(emails.shape)

#Data extraction
#print(emails.loc[1]['message'])

message = emails.loc[1]['message']
e = email.message_from_string(message)
#print(e.items())

#print(e.get('Date'))

#print(e.get_payload())

def get_filed(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))

    return column


#Extract headers(Only from file for now)
emails['From'] = get_filed("From", emails['message'])
print(emails.head(3))

emails['To'] = get_filed("To", emails['message'])
print(emails.head(2))


#Extract message body
def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column

emails['body'] = body(emails['message'])
#print(emails.head(2))

column_drop = ['file', 'message']
emails.drop(column_drop, axis=1, inplace=True)

#print(emails.head(2))

emails.to_csv("cleaned_email.csv", index=False)'''


df = pd.read_csv(r'C:\Users\Nmachi\Desktop\PycharmProjects\test\cleaned_email.csv')
#print(df.head(5))

data = df.body
d = data.to_numpy()
#print(type(d))
print(d.size)

for email in d:
    userEmail = email.split('.')
print(userEmail)



vocab_size = 100000
embedded_emails = [one_hot(sent, vocab_size) for sent in userEmail]
print(embedded_emails)


word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(userEmail, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
#print(length_long_sentence)

paddedEmail = pad_sequences(embedded_emails, length_long_sentence, padding='post')
#print(paddedEmail.size)
dfx = pd.DataFrame(paddedEmail)
X = dfx.drop([1,2,3,4,5,6,7,8,9,10,11,12])


target = df.From
labels = target.to_numpy()


for ids in labels:
    print(ids)


#One_hot encode the label
encoded_labels = [one_hot(d, vocab_size) for d in ids]
#print(encoded_labels)
y = pad_sequences(encoded_labels, maxlen=length_long_sentence, padding='post')


# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=length_long_sentence))
model.add(Flatten())
model.add(Dense(85, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y,  epochs=10, verbose=1)

loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: %f' % (accuracy*100))




