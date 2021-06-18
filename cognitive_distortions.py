import PySimpleGUI as sg
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

#sg.Window(title="Hello World", layout=[[]], margins=(250, 250)).read()

df = pd.read_csv('C:/Users/Mai/allnumbered.csv', names=['sentence', 'label', 'source'])
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
tokenizer  = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(df['sentence'])
sequences =  tokenizer.texts_to_sequences(df['sentence'])
word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

layout = [[sg.Text('Enter a sentence to be classified'), sg.Text(size=(15,1), key='-OUTPUT-')],      
                 [sg.InputText(key='-IN-')],      
                 [sg.Button('Enter'), sg.Exit()]]      

window = sg.Window('Cognitive distortions detection and classification', layout)    

def prepare(text):
    test_sentence = [text]
    tokenizer.fit_on_texts(test_sentence)
    test_sequences =  tokenizer.texts_to_sequences(test_sentence)
    print(test_sequences)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(test_data)
    return test_data

model = keras.models.load_model("C:/Users/Mai/LSTM2")
Categories = ["", "nondistorted", "Overgeneralization", "Should statement"]

#separate the sentences by . and put each one in a list, all lists and in a list to predict


while True:                             # The Event Loop
    event, values = window.read() 
    print(event, values)   
    #add the model and get output  
    prepared_text = prepare(values['-IN-'])  
    result = np.argmax(model.predict(prepared_text), axis=-1)
    category = Categories[result[0]]
    print(np.argmax(model.predict(prepared_text), axis = -1))
    if event == sg.WIN_CLOSED or event == 'Exit':
        break 
    if event == 'Enter':
        window['-OUTPUT-'].update(category)     

window.close()
