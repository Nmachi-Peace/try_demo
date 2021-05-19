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

import pandas as pd
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
import numpy as np

df = pd.read_csv ('C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\email_body.txt', sep='delimiter', header=None, engine='python')
df.columns=['body']
df.to_csv(r'C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\wholeData.csv', index=False)
#print(df.head(2))
#print(type(df))
#print(df.shape)

df_new = df[df['body'].notnull()]
data = df_new["body"].to_numpy().tolist()

#for email in data:
    #mssg = email.split('.')# converted the email body to a list
#print(mssg.)

vocab_size = 100000
#Encoded the email body using ont-hot encoding
encoded_emails = [one_hot(d.lower(), vocab_size) for d in data]
#print(encoded_emails)

#word_toknised to identify the sentecnce length
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(data, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
#print(length_long_sentence)


#Pad the sequences with the defined maximum length of the sequences which is 35
x = pad_sequences(encoded_emails, maxlen=length_long_sentence, padding='post')
#print(x)
#print(x.shape)


#X= x.reshape(1, -1)
#print(X.shape)




dataLabel = pd.read_csv ('C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\from_email_list.txt', sep='delimiter', header =None, engine='python')
dataLabel.columns=['from']
dataLabel.to_csv(r'C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\dataLabels.csv', index=None)
#print(dataLabel.head(2))

labels = dataLabel.to_numpy().tolist()
for ids in labels:
    '''print(ids)'''



encoded_labels = [one_hot(d, vocab_size) for d in ids]
#print(encoded_labels)
y = pad_sequences(encoded_labels, maxlen=length_long_sentence, padding='post')
#print(y)
#print(y.shape)

trainX, testX, trainy, testy = train_test_split(x, train_size=0.5, test_size=0.5, random_state=42, shuffle=True)







