#!/usr/bin/env python
# -*-  coding: UTF-8 -*-
# author: Powen Ko      www.powenko.com
import glob  # 導入 glob 模組，用於處理檔案路徑模式
import pickle  # 導入 pickle 模組，用於序列化和反序列化資料
import numpy  # 導入 numpy，用於處理數值運算
from music21 import converter, instrument, note, chord  # 導入 music21，用於 MIDI 分析和處理
from tensorflow.keras.models import Sequential  # 導入 Sequential 模型
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation  # 導入所需層
from tensorflow.keras.utils import to_categorical  # 用於將數據轉換為 one-hot 編碼
from tensorflow.keras.callbacks import ModelCheckpoint  # 用於保存最佳模型的回調函數

def train_network():
    """ 訓練一個神經網路以生成音樂 """
    notes = get_notes()  # 獲取音符資料
    print(notes)         # 印出所有音符

    t1=set(notes)        # 抓出所有的音符 並且去除重複的音符
    print(t1)       
    n_vocab = len(t1)    # 獲取獨特音符的數量  mapping tables
    print("有多少的音符 n_vocab=",n_vocab)
    network_input, network_output = prepare_sequences(notes, n_vocab)  # 準備序列資料

def get_notes():
    """ 從 ./midi_songs 資料夾的 MIDI 檔案中提取所有音符和和弦 """
    notes = []  # 初始化音符清單

    try:
        notes = pickle.load(open("data/notes", "rb"))  # 嘗試從檔案中讀取音符
    except:
        for file in glob.glob("midi_songs/*.mid"):  # 遍歷 midi_songs 資料夾中的所有 .mid 檔案
            midi = converter.parse(file)  # 解析 MIDI 檔案
            print("Parsing %s" % file)  # 印出正在解析的檔案名稱
            notes_to_parse = None  # 初始化要解析的音符變數
            try:
                s2 = instrument.partitionByInstrument(midi)  # 將音樂分為不同的樂器部分
                notes_to_parse = s2.parts[0].recurse()  # 取得第一部分的所有音符
            except:
                notes_to_parse = midi.flat.notes  # 若無樂器部分則直接使用平坦結構的音符

            for element in notes_to_parse:
                if isinstance(element, note.Note):  # 如果元素是音符
                    notes.append(str(element.pitch))  # 將音高加入音符清單
                elif isinstance(element, chord.Chord):  # 如果元素是和弦
                    notes.append('.'.join(str(n) for n in element.normalOrder))  # 將和弦轉為音高序列

        with open('data/notes', 'wb') as filepath:  # 將音符資料保存至檔案
            pickle.dump(notes, filepath)  # 序列化並寫入檔案

    return notes  # 返回音符資料

def prepare_sequences(notes, n_vocab):
    sequence_length = 5                 # 定義序列長度  五顆星

    pitchnames = sorted(set(item for item in notes))  # 獲取所有唯一的音符名稱

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))  # 建立音符對應整數的字典
    print("印出對照表字典", note_to_int)  # 印出音符對應整數的字典

    network_input = []  # X初始化輸入序列清單
    network_output = [] # Y初始化輸出序列清單

    t2=len(notes) - sequence_length                 # 計算所有音符的數量
    for i in range(0, t2, 1):                       # 遍歷所有音符，創建序列
        sequence_in = notes[i:i + sequence_length]  # (X)取得輸入序列     notes[0:100], notes[1:101] notes[2:102]
        sequence_out = notes[i + sequence_length]   # (Y)取得相應的輸出音符 notes[101],  notes[102],  notes[103]
        in1 = [note_to_int[char] for char in sequence_in]  # 將音符依照 對照字典 轉換為整數
        out1 = note_to_int[sequence_out]  # 將輸出音符 對照字典 轉換為整數
        network_input.append(in1)                   # 加入輸入清單
        network_output.append(out1)                 # 加入輸出清單

    n_patterns = len(network_input)  # 計算輸入樣本的數量
    print("一共產生多少筆的訓練資料 n_patterns=",n_patterns)  # 印出樣本數量

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))  # 調整輸入資料形狀
    print("network_input=", network_input)  # 印出輸入資料
    print("network_output=", network_output)  # 印出輸出資料

    network_input = network_input / float(n_vocab)  # 將輸入資料正規化
    network_output = to_categorical(network_output)  # 將輸出資料轉為 one-hot 編碼

    return (network_input, network_output)  # 返回準備好的輸入和輸出資料

if __name__ == '__main__':
    train_network()  # 執行訓練流程