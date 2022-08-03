import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df_train = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\MS\\NLP\\NlPs\\prediction\\data\\training_twitter_x_y_train.csv')
df_test = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\MS\\NLP\\NlPs\\prediction\\data\\test_twitter_x_test.csv')

a = df_train.airline_sentiment=='negative'
b = df_train.airline_sentiment =='positive'
a = np.array(a)
b = np.array(b)
df_train = df_train[a|b]

x_train = df_train['text'].str.lower()
y_train = df_train['airline_sentiment']


tokenizer = Tokenizer( num_words=5000,split=' ')

tokenizer.fit_on_texts(x_train)

vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(x_train)

X = pad_sequences(sequences, padding='post')

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y_train)))

embed_dim = 128
lstm_out = 256
model = Sequential()
model.add(Embedding(vocab_size, embed_dim,input_length = 33))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(vocab_size,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
model.fit(X, y, epochs =1 ,batch_size = 500)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

