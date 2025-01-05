import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import librosa
from tensorflow.keras import layers, models
"""
https://www.tensorflow.org/datasets/community_catalog/huggingface/speech_commands
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "bed",
            "bird",
            "cat",
            "dog",
            "happy",
            "house",
            "marvin",
            "sheila",
            "tree",
            "wow",
            "backward",
            "forward",
            "follow",
            "learn",
            "visual",
            "_silence_"
"""
# 目標標籤

TARGET_LABELS = ["yes", "no","up",
            "down",
            "left",
            "right","_silence_"] #"background_noise"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TARGET_LABELS)}

# 加載 Speech Commands 數據集
dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# 過濾目標標籤
def filter_target_labels(audio, label):
    label_str = tf.py_function(lambda x: info.features['label'].int2str(int(x)), [label], tf.string)
    return tf.reduce_any([label_str == target for target in TARGET_LABELS])

filtered_train_data = train_data.filter(filter_target_labels)
filtered_test_data = test_data.filter(filter_target_labels)

# 將標籤轉換為索引
def preprocess_label(audio, label):
    label_str = tf.py_function(lambda x: info.features['label'].int2str(int(x)), [label], tf.string)
    label_idx = tf.py_function(lambda x: LABEL_TO_INDEX[x.numpy().decode("utf-8")], [label_str], tf.int32)
    label_idx = tf.ensure_shape(label_idx, [])  # 確保標籤形狀為標量
    return audio, label_idx

filtered_train_data = filtered_train_data.map(preprocess_label)
filtered_test_data = filtered_test_data.map(preprocess_label)

# 音頻特徵提取
def preprocess_audio(audio, label):
    def extract_features(y):
        sr = 16000
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        max_length = 128  # 固定時間步長
        if mel_spec.shape[1] > max_length:
            mel_spec = mel_spec[:, :max_length]
        else:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, max_length - mel_spec.shape[1])), mode='constant')
        return mel_spec

    audio = tf.squeeze(audio)
    audio = tf.cast(audio, tf.float32) / 32768.0
    mel_spec = tf.numpy_function(extract_features, [audio], tf.float32)
    mel_spec = tf.ensure_shape(mel_spec, [40, 128])
    mel_spec = tf.expand_dims(mel_spec, axis=-1)
    return mel_spec, label

train_dataset = filtered_train_data.map(preprocess_audio).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = filtered_test_data.map(preprocess_audio).batch(32).prefetch(tf.data.AUTOTUNE)


# 模型構建
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(TARGET_LABELS), activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# 評估 filtered_test_data

# 使用模型進行評估
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"測試數據集的損失: {test_loss}")
print(f"測試數據集的準確率: {test_accuracy}")

# test filtered_test_data
# 測試單個樣本
for audio, label in filtered_test_data.take(1):  # 提取單個樣本
    # 處理音頻特徵
    audio, label = preprocess_audio(audio, label)
    audio = tf.expand_dims(audio, axis=0)  # 增加批次維度
    predictions = model.predict(audio)

    # 預測結果
    predicted_label = TARGET_LABELS[np.argmax(predictions)]
    true_label = TARGET_LABELS[label.numpy()]
    print(f"真實標籤: {true_label}, 預測標籤: {predicted_label}")

# 保存模型
model.save("speech_recognition_model.keras")


##### 載入模型
model = tf.keras.models.load_model('speech_recognition_model.keras')   # 載入已保存的模型

# 保存模型權重
model.save_weights("model.weights.h5")           # 保存模型的權重到檔案 model.weights.h5

# 讀取模型權重
model.load_weights("model.weights.h5")           # 從 model.weights.h5 檔案讀取模型權重