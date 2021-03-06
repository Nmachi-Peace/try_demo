
from numpy import array
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import  Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd


email=[
    'kay.mann@enron.com',
    'We are in a loop.  We need a process to deal with this.  All ideas welcome.',
    'I\'m looking for ISDA\'s between Enron MW and ENA and RMT.  I don\'t know where to look.  Can someone locate these for me?\nThanks,',
    'Could I get a copy of the Midland Cogeneration Venture Limited Partnership termination letter? \n Thanks,',
    'This is an internal draft, which has not been sent to CRRA, as far as I know. \n Kay',
    'FYI.  I\'ll bring you the ONSI confidentiality agreement.',
    'You try and see if he notices.',
    'I would like to take Dec 26, 31 and Jan. 2 as vacation days.\n Thanks,\n Kay"',
    'I\'m in the Enron Basics of Risk Management seminar Thurs (til 5) and Fri am.\nI would like to take next Wednesday as a day of vacation.\nThanks,\nKay"',
    'I know there\'s a lot of anxiety about severance, reflecting the possibility that the existing Enron severance plan might be changed. I noticed in the merger',
    'agreement that a ""floor"" was set on severance.  Have you heard anything about what Dynegy has for severance?',
    'I know there are exceptions, but I was just curious, as lots of folks are very nervous.',
    'From the merger agreement:',
    'I can do that, especially if I get the list early in the day.',
    'Jeff,\n What\'s your suggestion?  We are looking at a PA in connection with Tennessee Pipeline\'s open season. (did I get that right?)',
    'Stephanie Panus has a few extra copies at her Desk.',
    'At the very least, the cancellation charges are theirs.',
    'Being revised currently, but lays out the deal:',
    'vince.kaminski@enron.com',
    'Jim,\nThanks a lot. It is difficult to find a better example example of commitment',
    'into Enron and to professional excellence\nthan  our weather guys.\n\nVince',
    'We shall forward to you shortly a copy of the message from Sandeep with the\nnumbers you have ',
    'requested.  What follows below are the extracts from two recent articles on',
    'the power situation in India\npublished by The Financial Times.',
    'The first article describes recent power outage in northern India affecting\nmillions of people.',
    'One possible line of defense is pointing out to the value of  output lost due\nto  power',
    'shortages. It is obvious that the value of lost production exceeds the cost\nof power produced',
    'by the Dhabol plant (expensive as it may be). The power cut affected 230\nmillion people.',
    'The second article is Enron specific.\nVince',
    'Kim,\nFYI. I checked on the progress of the accounting adjustment and was told it would happen\nthis month.\n\nVince"',
    'Greg,\n\nI am forwarding you the notes from a regular weekly meeting with one of our\nsummer interns.',
    'He works on review of the literature regarding drivers behind bid - offer\nspreads.',
    'Please, let me know if you would like additional info.\n\nVince',
    'Greg,\n\nI scheduled a meeting with Norman Packard from the Prediction Company\nin Houston, July the 30th, 9:00 a.m.',
    'Liz blocked out a few hours for the meeting.\n\nVince"',
    'Greg,\n\nThis is the info  about the  WEBI program at Wharton\n\n.Vince',
    'Any interest in this conference?\n\nhttp://www.haas.berkeley.edu/citm/conferences/010522/\n\n Vince',
    'jeff.dasovich@enron.com',
    'Since Congressman\'s Ose\'s been asking FERC alot of questons about what the ISO, DWR, etc. have been up to, perhaps he could write a ""Dear FERC"" letter saying',
    '""here\'s what I\'m hearing, it sounds serious and you need to investigate, pronto, and get back to me ASAP with your conclusions.""\n\nBest\nJeff',
    'Tim/Bob:\n\nAttached is the letter that we sent to Lynch explaining the info we thought ',
    'ought to be made publicly available.  We\'re discussing how we can ensure that ',
    'the market has access to any and all information that FERC might use in its',
    '""investigation"" so that independent analyses might be undertaken.  Is there',
    'anything in addition to the information we included in the letter to Lynch',
    '(or that is already be publicly available) that FERC might use in its ',
    'investigation and that we ought to target for public release?  Thanks alot.\n\nBest,\nJeff',
    'FYI.  The report posted on the website fails to include the report\'s',
    'transmittal letter.  I just found that letter in the mail.  The letter ',
    'explains that the report\'s analysis DOES NOT include ""the events"" of May,',
    'June and July, which the Compliance unit is currently studying.  It will ',
    'release the results of that study in the Fall.',
    'Jennifer:\nI\'m on a call with Delainey that just got convened, which means that I might ',
    'be delayed till 11:45.  Unfortunately, Harry\'s on vacation.  I\'ll call in ',
    'just as soon as I get done with Delainey.  Also, I was on a call with Eric',
    'yesterday, during which I went over the S.D. legislation, so he should be',
    'fairly up to speed.  I\'ll get on just as soon as I can.\n\nBest,\nJeff'
]

#print(len(email))

# define class labels
labels = array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])

from nltk.tokenize import word_tokenize
all_words = []
for sent in email:
    tokenise_word = word_tokenize(sent)
    for word in tokenise_word:
        all_words.append(word)

#print(len(all_words))

unique_words = set(all_words)
#print(len(unique_words))

dim = 20
vocab_size = 10000
embedded_emails = [one_hot(data, vocab_size) for data in email]
#print(embedded_emails)
#print(len(embedded_emails))

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(email, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))

padded_sentences = pad_sequences(embedded_emails, length_long_sentence, padding='post')
#print(padded_sentences)


model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=length_long_sentence))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(padded_sentences, labels, epochs=10, verbose=2)

loss, accuracy = model.evaluate(padded_sentences, labels, verbose=2)
print('Accuracy: %f' % (accuracy*100))

'''
with open('stylo_project.txt') as f:
    data = f.read()
    #print(data)

from nltk.tokenize import word_tokenize
all_words = []
for sent in data:
    tokenise_word = word_tokenize(sent)
    for word in tokenise_word:
        all_words.append(word)

unique_words = set(all_words)
#print(len(unique_words))


embedded_emails = [one_hot(data, 3) for data in labels ]

print(embedded_emails)


max_length = 40
padded_sentences = pad_sequences(embedded_emails, maxlen=max_length, padding='post')
print(padded_sentences)

df = pd.DataFrame(padded_sentences[labels])

print(df)

x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.33, random_state=4)



print(x_train.shape)
print(y_test.shape)

dim = 10
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(x_train, y_test, epochs=150, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(x_train, y_test, verbose=2)
print('Accuracy: %f' % (accuracy*100))
'''




