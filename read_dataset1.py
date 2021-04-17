from numpy import array
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
import email

from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import word_tokenize




'''
#Define the main data
emails = pd.read_csv("C:\\Users\\Nmachi\\Desktop\\PycharmProjects\\test\\emails.csv")
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
#print(emails.head(3))


#Extract message body
def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column

emails['body'] = body(emails['message'])
print(emails.head(2))

column_drop = ['file', 'message']
emails.drop(column_drop, axis=1, inplace=True)

print(emails.head(2))

emails.to_csv("cleaned_email.csv", index=False)'''

#Define the cleaned data
df = pd.read_csv(r'C:\Users\Nmachi\Desktop\PycharmProjects\test\cleaned_email.csv')
#print(df)

#print(df.head(1000))

#print(df.isnull() .sum())

#print(df.describe())

#print(df.corr())


#define vocalbulary size
vocab_size = 10000

#Define sequence length
max_length = 35

#Extract the email body and convert it to a numpy array
data = df["body"].to_numpy()
for email in data:
    Xmsg = email.split('.')# converted the email body to a list
#print(Xmsg)

#Encoded the email body using ont-hot encoding
encoded_emails = [one_hot(d, vocab_size) for d in Xmsg]
#print(encoded_emails)

#word_toknised to identify the sentecnce length
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(Xmsg, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
print(length_long_sentence)


#Pad the sequences with the defined maximum length of the sequences which is 35
x = pad_sequences(encoded_emails, maxlen=length_long_sentence, padding='post')
#print(x)
#print(x.shape)

#Data from x in order to drop some rows from the table so to avoid the value error of unequal sample between x and y
dfx = pd.DataFrame(x)
X = dfx.drop([1,2,3,4,5,6,7,8,9,10,11,12])#droped 12 rows to have (23, 35) and (23,35) for X and y
print(X.shape)


#Extract email From  and convert it to a numpy array
dlabels = df["From"].to_numpy()
#print(dlabels)

for labels in dlabels:
    '''print(labels)'''

#One_hot encode the label
encoded_labels = [one_hot(d, vocab_size) for d in labels]
#print(encoded_labels)

#Pad the lavbels
y = pad_sequences(encoded_labels, maxlen=length_long_sentence, padding='post')
#print(y.shape)

#print(y.shape)


#tokenised = Tokenizer(num_words=1000, oov_token='OOV')
#x_tokenised = tokenised.fit_on_texts(Xmsg)
#print(x_tokenised)



#X = x_tokenised.texts_to_sequences(Xmsg)
#word_index = tokenised.word_index


#tokenised = Tokenizer(num_words=1000, oov_token='OOV')
#tokenised.fit_on_texts(Y)
#y_tokenised = tokenised.texts_to_sequences(Y)


#Using Train_test_split function for training and testing and split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, random_state=True, shuffle=True)

#check the shape of X and y
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(X_train.shape)


vocab_size = 10000
dim = 10

# define the model
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=length_long_sentence))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dropout(rate=0.4))
model.add(Dense(85, activation='softmax'))
model.add(Dropout(rate=0.4))

#print(len(model.weights))
# compile the model
#sgd = SGD(lr=0.1, momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# summarize the model
print(model.summary())

# fit the model
model.fit(X_train, Y_train,  epochs=10, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))



