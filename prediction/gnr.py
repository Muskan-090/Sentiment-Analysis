import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, model_from_json
from . import Nps


df_train = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\MS\\NLP\\NlPs\\prediction\\data\\training_twitter_x_y_train.csv')
# df_test = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\MS\\NLP\\NlPs\\prediction\\data\\test_twitter_x_test.csv')

# a = df_train.airline_sentiment=='negative'
# b = df_train.airline_sentiment =='positive'
# a = np.array(a)
# b = np.array(b)
# df_train = df_train[a|b]

# x_train = df_train['text'].str.lower()
# y_train = df_train['airline_sentiment']


# tokenizer = Tokenizer( num_words=5000,split=' ')

# tokenizer.fit_on_texts(x_train)

# vocab_size = len(tokenizer.word_index) + 1

# sequences = tokenizer.texts_to_sequences(x_train)

# X = pad_sequences(sequences, padding='post')

# y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y_train)))

# embed_dim = 128
# lstm_out = 256
# model = Sequential()
# model.add(Embedding(vocab_size, embed_dim,input_length = 34))
# model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(vocab_size,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
# # print(model.summary())
# model.fit(X, y, epochs =3 ,batch_size = 500)
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")




def Movie_reviews(txt):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(Nps.X, Nps.y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        
    
    text = Nps.tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(text,33)
    predicted = np.argmax(loaded_model.predict(padded),axis = -1)

    positive = len(predicted[predicted==1])
    negative = len(predicted[predicted==0])
    if positive>negative:
        return "positive"
        
    else:
        return  "negative"
    
    




