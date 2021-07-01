import  os
from email.parser import Parser



'''
rootdir = "C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\dest"
for directory, subdirectory, filenames in os.walk(rootdir):
    print(directory, subdirectory, len(filenames))


dir = "C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\dest\\lay-k"
def email_analyse(inputfile, to_email_list, from_email_list, email_body):
    with open(inputfile, "r") as f:
        data = f.read()
    email = Parser().parsestr(data)

    if email['to']:
        email_to = email['to']
        email_to = email_to.replace("\n", "")
        email_to = email_to.replace("\t", "")
        email_to = email_to.replace(" ", "")

        email_to = email_to.split(",")

        for email_to_1 in email_to:
            to_email_list.append(email_to_1)

    from_email_list.append(email['from'])
    email_body.append(email.get_payload())


all_to_email_list = []
all_from_email_list =[]
all_email_body = []

for directory, subdirectory, filenames in os.walk(rootdir):
    for filename in filenames:
        email_analyse(os.path.join(directory, filename), all_to_email_list,  all_from_email_list, all_email_body)


with open("to_email_list.txt", "w") as f:
    for to_email in all_to_email_list:
        if to_email:
            f.write(to_email)
            f.write("\n")

with open("from_email_list.txt", "w") as f:
    for from_email in all_from_email_list:
        if from_email:
            f.write(from_email)
            f.write("\n")

with open("email_body.txt", "w") as f:
    for email_bod in all_email_body:
        if email_bod:
            f.write(email_bod)
            f.write("\n")
'''

import email
import shutil
import glob
import pandas as pd
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import re
import email
from nltk.tokenize import word_tokenize
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from os import makedirs
from keras.models import load_model
import numpy as np
import neattext.functions as nfx

#df = pd.read_csv ('C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\email_body.txt', sep='delimiter', header=None, engine='python')
#df.columns=['body']
#df.to_csv(r'C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\wholeData.csv', index=False)
#print(df.head(2))
#print(type(df))
#print(df.shape)




df = pd.read_csv('combined_emailBody.csv', error_bad_lines=False)
#print(df.shape())

dlabel = pd.read_csv('combined_labels.csv')
#print(dlabel.shape)


new_df = df.replace(to_replace='None', value=np.nan).dropna()
#new_df = df.dropna(inplace=True)
#print(type(new_df))

dataEmail = new_df["body"].to_numpy()

for e in dataEmail:
    splitted_email = e.split('.')  # converted the email body to a list
    #print(email)

#word toknised to identify the sentecnce length
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(splitted_email, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
#print(length_long_sentence)

#Tokenize the sentences
tokenizer = Tokenizer()
#preparing vocabulary
tokenizer.fit_on_texts(list(dataEmail))

#converting text into integer sequences
tokenised_email = tokenizer.texts_to_sequences(dataEmail)
#print(tokenised_email)

voc_size = len(tokenizer.word_index) + 1 #+1 for padding
#print("check the voc size: ", voc_size)

x = pad_sequences(tokenised_email, maxlen=length_long_sentence, padding='post')
#print(x.shape)

x_df = pd.DataFrame(x)
#print(x_df.shape)

X = x_df.iloc[0-601305:, :]

#print(X.shape)

label = dlabel['From'].to_numpy()
#print(labels)

tokenizer = Tokenizer()
#preparing label
tokenizer.fit_on_texts(list(label))
tokenisedLabel = tokenizer.texts_to_sequences(label)
y = pad_sequences(tokenisedLabel, maxlen=length_long_sentence, padding='post')
#print(y.shape)

y_df = pd.DataFrame(y)
#print(y_df.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.33, random_state=42, shuffle=True)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)


# define the model
model = Sequential()
model.add(Embedding(voc_size, 8, input_length=length_long_sentence))
model.add(Flatten())
model.add(Dense(85, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,  epochs=10, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


#yy = y.reshape(5, 83904)
#print(yy.shape)

#X = x.reshape(5, 3053286)
#print(X.shape)

#X = x.transpose()
#Y = y.transpose()

#print("The X ", X.shape, "and Y ", Y.shape)


'''
dim = 2
modelen = Sequential()
modelen.add(Embedding(voc_size, dim, input_length=length_long_sentence))
modelen.add(Flatten())
#modelen.add(Dropout(rate=0.4))
modelen.add(Dense(85, activation='softmax'))
# compile the model
modelen.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize the model
print(modelen.summary())
# fit the model
modelen.fit(x, y, epochs=100, verbose=1)

#histtory = modelen.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1)
# evaluate the model
# evaluate the model
#_, train_acc = modelen.evaluate(X_train, y_train, verbose=0)
#_, test_acc = modelen.evaluate(X_test, y_test, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
'''







