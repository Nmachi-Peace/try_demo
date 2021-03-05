import io
import os
import re
import shutil
import string
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt


from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



dataset = "C:\\Users\\Nmachi\\Desktop\\enron1"

dataset_dir = os.path.join(os.path.dirname(dataset), 'C:\\Users\\Nmachi\\Desktop\\enron1')
#print(os.listdir(dataset_dir))


batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'C:\\Users\\Nmachi\\Desktop\\enron1', batch_size=batch_size, validation_split=0.2,
    subset = 'training', seed = seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'C:\\Users\\Nmachi\\Desktop\\enron1', batch_size=batch_size, validation_split=0.2,
    subset = 'validation', seed=seed)

for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy(), text_batch.numpy()[i])

Autotune = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=Autotune)
val_ds = val_ds.cache().prefetch(buffer_size=Autotune)

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

result = embedding_layer (tf.constant([1, 2, 3]))
#print(result.numpy())

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
#print(result.shape)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '\r\r', ' ')
    return tf.strings.regex_replace(stripped_html,
                                     '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 100


vectorise_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

text_ds = train_ds.map(lambda x, y: x)
vectorise_layer.adapt(text_ds)

embedding_dim = 16
model = Sequential([
    vectorise_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)])

#logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])


model.summary()

print(history.history)
print(history.history.keys())
#Accuracy graph
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc ='upper left')
plt.show()


#Loss graph
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc ='upper right')
plt.show()

#Vaal-accuracy graph
plt.plot(history.history['val_accuracy'])
plt.title('model val_accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.legend(['val'], loc ='upper left')
plt.show()

#val-loss graph
plt.plot(history.history['val_loss'])
plt.title('model val_loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['val'], loc ='upper right')
plt.show()



