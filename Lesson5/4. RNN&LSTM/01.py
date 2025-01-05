import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 1. 生成時間序列數據
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # 正弦波1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # 正弦波2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # 噪聲
    return series[..., np.newaxis]

# 參數
n_steps = 50  # 每個序列的步長
batch_size = 1000

# 生成訓練和測試數據
series = generate_time_series(batch_size, n_steps + 1)
X_train, y_train = series[:700, :n_steps], series[:700, -1]
X_valid, y_valid = series[700:900, :n_steps], series[700:900, -1]
X_test, y_test = series[900:, :n_steps], series[900:, -1]

# 2. 定義 RNN 模型
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),  # 第一層 RNN
    tf.keras.layers.SimpleRNN(20),                                              # 第二層 RNN
    tf.keras.layers.Dense(1)                                                    # 全連接層
])

# 編譯模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. 訓練模型
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# 4. 評估模型
mse, mae = model.evaluate(X_test, y_test)
print(f"測試集 MSE: {mse}, MAE: {mae}")

#%% 
# 5. 預測並繪圖
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="真實值", marker='o')
plt.plot(y_pred, label="預測值", marker='x')
plt.legend()
plt.xlabel("測試樣本索引")
plt.ylabel("值")
plt.title("RNN 預測 vs 真實值")
plt.show()

# %%
