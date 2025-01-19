# 匯入必要的模組
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # 這是 TensorFlow Text 的相依套件

# 定義 BERT 模型與預處理的 URL
bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

# 建立 BERT 的預處理和模型層
bert_preprocess_layer = hub.KerasLayer(bert_preprocess_url)
bert_model_layer = hub.KerasLayer(bert_model_url)

# 測試文本數據
texts = tf.constant(["I love TensorFlow!", "BERT is amazing!"])

# 預處理文本
preprocessed_texts = bert_preprocess_layer(texts)

# 通過 BERT 模型獲取輸出
bert_outputs = bert_model_layer(preprocessed_texts)

# 查看 BERT 的輸出
pooled_output = bert_outputs["pooled_output"]  # 用於分類任務
sequence_output = bert_outputs["sequence_output"]  # 用於序列標註任務

print("Pooled Output:", pooled_output.shape)
print("Sequence Output:", sequence_output.shape)

# 建立簡單的文本分類模型
def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessed_text = bert_preprocess_layer(text_input)
    outputs = bert_model_layer(preprocessed_text)
    pooled_output = outputs["pooled_output"]
    dropout = tf.keras.layers.Dropout(0.2)(pooled_output)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(dropout)

    model = tf.keras.Model(inputs=[text_input], outputs=[output])
    return model

# 編譯模型
model = build_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 模型摘要
model.summary()

# 假設的訓練數據
train_texts = ["This is great!", "I hate it.", "This is bad.", "I love it."]
train_labels = [1, 0, 0, 1]

# 訓練模型
history = model.fit(x=train_texts, y=train_labels, epochs=3)

# 測試模型
test_texts = ["I really like it!", "This is not good."]
predictions = model.predict(test_texts)
print("Predictions:", predictions)
