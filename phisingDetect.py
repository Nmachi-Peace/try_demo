import re
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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

data=[
    'David.Allan@ENRON.com',
    'Allen,',

    'First of all, dress code is business casual (khakis etc.). My assistant,',
    'Nyree Chanaba, will get back to you with particulars on the hotel. It',
    'will probably be the Hyatt, which is kitty-corner from our office. But',
    'whichever hotel it is, the concierge will be able to direct you to the',
    'Enron Building: it\'s well-known, and we\'ll have you within 2-3 blocks of',
    'our office at the most.',

    'The morning session is for quiet conversations with anybody who wants',
    'one-on-one time with you. At this point I don\'t know whether that will',
    'be a lot of people or a few (we\'re in the middle of vacation time so',
    'it\'s hard to predict attendance), but I would say 9 a.m. would be good.',
    'Go to the lobby of the building and ask the security guards to call me',
    'or Nyree (we\'re on the 29th floor).',

    'If it\'s OK with you, why don\'t you pay your hotel bill and then fax me',
    'the bill when you get home on Friday so we can cut a check.',

    'If you\'re not too tired of Enron people at that time, I\'d like to take',
    'you out to dinner on Thursday night.',

    'As to the projector, we\'ll have something set up; in case the connection',
    'doesn\'t work we also have laptops we can hook up.',

    'We\'ll communicate more as the week goes on. We look forward to seeing',
    'you!',

    'Dave',
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
    'anything in addition to the information we included in the letter to Lynch ',
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
    'fairly up to speed.  I\'ll get on just as soon as I can.\n\nBest,\nJeff',
    'phillip.allen@enron.com',
    '"Traveling to have a business meeting takes the fun out of the trip.',
    'Especially if you have to prepare a presentation.',
    'I would suggest holding the business plan meetings here then take a trip without any formal business meetings.',
    'I would even try and get some honest opinions on whether a trip is even desired or necessary.',
    'As far as the business meetings, I think it would be more productive to try and stimulate discussions across the different groups about what is working and what is not.',
    'Too often the presenter speaks and the others are quiet just waiting for their turn.',
    'The meetings might be better if held in a round table discussion format.',
    'My suggestion for where to go is Austin.',
    'Play golf and rent a ski boat and jet ski\'s.',
    'Flying somewhere takes too much time."',
    'Traveling to have a business meeting takes the fun out of the trip.  Especially if you have to prepare a presentation.  I would suggest',
    'holding the business plan meetings here then take a trip without any formal business meetings.  I would even try and get some honest',
    'opinions on whether a trip is even desired or necessary.',

    'As far as the business meetings, I think it would be more productive to try and stimulate discussions across the different groups about',
    'what is working and what is not.  Too often the presenter speaks and the others are quiet just waiting for their turn.   The meetings',
    'might be better if held in a round table discussion format.',

    'My suggestion for where to go is Austin.  Play golf and rent a ski boat and jet ski\'s.  Flying somewhere takes too much time.',

    'laura.scott@enron.com,',
    '"Congratulations to you as well Sally!  It is very encouraging to see a female Managing Director.',
    'All the best.',
    'shona.wilson@enron.com',
    '"Dear Bill,',
    'I am currently helping Milind Laad and Heidi Hellman establish a risk',
    'operations department for the India trading company.',
    'I was wondering if you would have 10 minutes to meet with me sometime this week to discuss how you',
    'see India moving forward and in what time frame as this would help me a great',
    'deal in determining what types of systems, processes, and controls we should set up.',

    'Thanks,',

    'Shona"',
    'mike.jordan@enron.com,',
    '"To all',
    'Tim has delivered the initial microsoft project plan for the MO integration - ',
    'this is available to all via the integration project \'office\' - i.e. Richard and Phil.',
    'cheryl.kuehl@enron.com',
    '"Please notate on your calendar:',
    'The room location for the Associate PRC at the Hyatt will Arboretum 4 and 5',
    'rather than Arboretum 1 and 2.',
    'Thank you for your participation."',
    'sara.shackleton@enron.com',
    'Corinne:',

    'Attached is your arbitration language (marked) and our arbitration text.  I ',
    'hope this does it!  Thanks.',
    'This is a ""high volume"" trading counterparty.  Shouldn\'t we negotiate an ISDA',
    'to replace the existing master?  SS"',
    'samon_corinne@jpmorgan.com',
    'Sara,',
    'Thank you for your fax.',
    'Section 180.3 of the CEA prohibits us from agreeing to any settlement',
    'procedure/forum ahead of time.  JPMFI, in accordance with 180.3, will offer 3',
    'forums - which will include AAA.  If Enron chooses AAA, the arbitration will',
    'be',
    'subject to the provisions specified in your fax dd 2/26/2001.',

    'Attached is JPM\'s proposed language for the arbitration agreement.',

    '(See attached file: ARBITRATION AGREEMENT.doc)',

    'If you are in agreement with the language, please forward Enron\'s proposed',
    'language (detailed in your 2/26 fax) electronically (if available).',

    'Thank you and Regards,',
    'Corinne',
    'shelley.corman@enron.com',
    'We had 22 kids and 3 adult helpers.  The kids were pretty wild.  It wasn\'t one of our better days.  Oh well!',
    'If it\'s okay with you, I\'d like to take a week break next weekend.  I need to be at an event downtown at 1 Pm and it would be nice not ',
    'to rush.  I will bring you the boom box and CDs with music suggestions.  Would that be okay?',
    'Also, Stan is not available the third weekend in November so I have agreed to switch with him.  He will teach Nov 11 (I will be there',
    'to help)  and I will take Nov 18 (Kim promises that he will definitely be there).',
    'monika.causholli@enron.com',
    'Hello Margaret,',


    'I don\'t know if you remember but I gave you my and my husband\'s passports to obtain H1b visas in Washington DC. You said that I have to ',
    'fill out a couple of forms before they were send out. Does that sound right to you or did I understand it wrong?',
    'Do you know how long it takes for the forms to be mailed? The reason I am asking this question is because I am afraid that the visa ',
    'process will be delayed and I have plans to travel in December. Without bothering you too much, can you explain to me the procedure.',
    'For example, Do I have to fill these two forms before the passports are mailed to DC or after. If I have to fill these forms before how',
    ' can I speed up the process?',

    'thank you,',
    'Monika Causholli',
    'aewhitman@mindspring.com',
    'Hi, David,',

    'A couple of questions:',

    'dress code for visit',
    'address and name of hotel.  My plane (DL 1857)is scheduled to arrive',
    'at',
    '1853.  I plan to go directly to the hotel.',
    'do you want me to pay the bill and rebill you when I send the',
    'invoice?',
    'directions from the hotel to your office?  When do you want me to be',
    'there on Thursday?',
    'I plan to bring my laptop with several ppt presentations to show.  Do',
    'you',
    'have a hookup for the laptop to a projector?  I 	plan to leave a',
    'copy of the',
    'ppt slides with you if you have a zip file that can download them from',
    'my',
    'laptop.',
    'I believe you were going to send me some info about Enron\'s plans and',
    'objectives, now that I am on the team and have 	signed the',
    'confidentiality',
    'agreements.',
    'any other questions I need to ask?',

    'See you on Thursday.',

    'Allen',
]


# define class labels
labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,
                   6,6,6,6,6,6,6,6,6,7,7,7,7,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
                   13,13,13])

#email_ID = re.findall('\S+@\S+', data)

#for ids in email_ID:
   #print(ids)



word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(data, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
#print(length_long_sentence)

#Tokenize the sentences
tokenizer = Tokenizer()
#preparing vocabulary
tokenizer.fit_on_texts(list(data))
#converting text into integer sequences
tokenised_email = tokenizer.texts_to_sequences(data)

voc_size = len(tokenizer.word_index) + 1 #+1 for padding
#print("check the voc size: ", voc_size)

X = pad_sequences(tokenised_email, maxlen=length_long_sentence, padding='post')
print(X.shape)

y = to_categorical(labels)
#print(label.shape)

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.4, random_state=2)
#print(trainX.shape, trainy.shape)

vocab_size = 10000
dim = 40
model = Sequential()
model.add(Embedding(vocab_size, dim, input_length=length_long_sentence))
model.add(Flatten(input_shape=[38]))
model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(100, activation="relu"))
#model2.add(Dropout(rate=0.4))
model.add(Dense(14, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
histtory = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=25, verbose=1)
# evaluate the model
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))




