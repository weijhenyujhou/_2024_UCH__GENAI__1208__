#!/usr/bin/env python
# -*-  coding: UTF-8 -*-
# author: Powen Ko      www.powenko.com
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint



def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    print(notes)
    # get amount of pitch names
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    try:
        notes = pickle.load(open("data/notes", "rb"))
    except:
        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)
            print("Parsing %s" % file)
            notes_to_parse = None
            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    #獲取所有音調名稱
    pitchnames = sorted(set(item for item in notes))
    # 創建一個音調對應到整數表
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    # 建立輸入序列和相應的輸出
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        in1=[note_to_int[char] for char in sequence_in]
        out1=note_to_int[sequence_out]
        network_input.append(in1)
        network_output.append(out1)
    n_patterns = len(network_input)
    # 將輸入資料大小調整，到LSTM圖層的輸入大小
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # 規範化輸入和輸出　
    print("network_input=",network_input)
    print("network_output=",network_output)
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    return (network_input, network_output)



def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    return model

if __name__ == '__main__':
    train_network()