import tensorflow as tf
import numpy as np

# 創建一個簡單的自編碼器模型
def create_autoencoder(input_dim, encoding_dim):
    # 編碼器部分
    input_data = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)

    # 解碼器部分
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

    # 創建自編碼器模型
    autoencoder = tf.keras.models.Model(input_data, decoded)

    # 編譯模型
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mse')


    return autoencoder

# 定義自編碼器參數
input_dim = 32  # 輸入特徵維度
encoding_dim = 16  # 編碼器的維度


# 創建一個示例數據集
data = np.random.rand(100, input_dim)  # 100個樣本，每個樣本有32個特徵


# 創建自編碼器模型
autoencoder = create_autoencoder(input_dim, encoding_dim)

# 訓練自編碼器模型
autoencoder.fit(data, data, epochs=10000, batch_size=100)

# 使用訓練後的模型進行數據重建
reconstructed_data = autoencoder.predict(data)

# 打印一些示例數據
print("原始數據:")
print(data[0])
print("重建數據:")
print(reconstructed_data[0])

# 打印編碼後的數據
encoder = tf.keras.models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
encoded_data = encoder.predict(data)
print("編碼後的數據:")
print(encoded_data[0])