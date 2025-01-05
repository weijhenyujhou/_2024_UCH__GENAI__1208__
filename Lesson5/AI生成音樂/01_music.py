#!/usr/bin/env python
# -*-  coding: UTF-8 -*-
# author: Powen Ko      www.powenko.com
# pip install music21
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

s = corpus.parse('bach/bwv65.2.xml')
s.analyze('key')
s.show()
"""

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    print(notes)


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



if __name__ == '__main__':
    train_network()