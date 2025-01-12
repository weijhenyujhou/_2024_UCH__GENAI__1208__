import tensorflow as tf  # 引入 TensorFlow 模組，用於深度學習
import numpy as np  # 引入 Numpy，用於數據處理
import PIL.Image  # 引入 PIL 模組，用於圖片處理
import matplotlib.pyplot as plt  # 引入 Matplotlib 用於顯示圖片

# 載入圖片的輔助函式
def load_image(path_to_img):
    max_dim = 512  # 設定圖片的最大邊長
    img = tf.io.read_file(path_to_img)  # 讀取圖片檔案
    img = tf.image.decode_image(img, channels=3)  # 解碼圖片，確保為 RGB 格式
    img = tf.image.convert_image_dtype(img, tf.float32)  # 將圖片轉為浮點數格式（0~1）
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # 獲取圖片的形狀，轉為浮點數
    long_dim = tf.reduce_max(shape)  # 計算圖片的長邊
    scale = max_dim / long_dim  # 計算縮放比例
    new_shape = tf.cast(shape * scale, tf.int32)  # 計算新的形狀，並轉為整數
    img = tf.image.resize(img, new_shape)  # 調整圖片大小
    img = tf.expand_dims(img, axis=0)  # 添加 batch 維度
    return img  # 返回處理後的圖片

# 顯示圖片的輔助函式
def show_image(image):
    image = np.array(image)  # 將 Tensor 轉為 Numpy 陣列
    plt.imshow(image)  # 顯示圖片
    plt.axis('off')  # 隱藏座標軸
    plt.show()  # 顯示圖片視窗

# 定義 DeepDream 模型
def deep_dream_model(layer_names):
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')  # 載入預訓練的 InceptionV3 模型（不包含頂層）
    layers = [base_model.get_layer(name).output for name in layer_names]  # 提取指定層的輸出
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)  # 建立新的模型，僅輸出指定層
    return dream_model  # 返回 DeepDream 模型

# 定義 DeepDream 處理
@tf.function
def deep_dream_step(img, model, step_size):
    with tf.GradientTape() as tape:  # 建立 GradientTape 用於梯度計算
        tape.watch(img)  # 監視輸入圖片
        outputs = model(img)  # 通過模型取得指定層的輸出
        loss = tf.reduce_sum([tf.reduce_mean(output) for output in outputs])  # 計算損失（輸出的平均值總和）
    grads = tape.gradient(loss, img)  # 計算圖片相對於損失的梯度
    grads /= tf.math.reduce_std(grads) + 1e-8  # 正規化梯度，避免數值過大或過小
    img += step_size * grads  # 更新圖片，放大模型的特徵偏好
    img = tf.clip_by_value(img, 0.0, 1.0)  # 限制圖片的像素值範圍在 0~1
    return loss, img  # 返回損失值和更新後的圖片

# 開始 DeepDream
def run_deep_dream(img, model, steps, step_size, dream_strength):
    for step in range(steps):  # 迭代多個步驟
        loss, img = deep_dream_step(img, model, step_size * dream_strength)  # 調整步伐大小以改變 Dream 強度
        print(f"Step {step+1}, Loss: {loss.numpy():.4f}")  # 輸出當前步驟的損失值
    return img  # 返回處理後的圖片

# 主程式
if __name__ == '__main__':
    image_path = '02.jpeg'  # 圖片路徑，替換為你的圖片檔案
    original_img = load_image(image_path)  # 載入圖片

    dream_layers = ['mixed3', 'mixed5']  # 定義目標層，用於放大特定特徵
    dream_model = deep_dream_model(dream_layers)  # 建立 DeepDream 模型

    dream_img = tf.identity(original_img)  # 複製原始圖片作為輸入
    steps = 100  # 定義處理步驟數
    step_size = 0.01  # 定義每步的梯度步伐大小
    dream_strength = 1.5  # 定義 Dream 的強度參數，可以調整為更大或更小

    # 執行 DeepDream
    dream_img = run_deep_dream(dream_img, dream_model, steps, step_size, dream_strength)

    # 顯示並保存結果
    show_image(dream_img[0])  # 顯示結果圖片
    result = PIL.Image.fromarray(np.uint8(dream_img[0] * 255))  # 將圖片轉為 8-bit 格式
    result.save('dream_result.jpg')  # 保存結果為檔案