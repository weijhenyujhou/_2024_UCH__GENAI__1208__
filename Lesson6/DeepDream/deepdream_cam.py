import cv2  # 引入 OpenCV 模組
import tensorflow as tf  # 引入 TensorFlow 模組
import numpy as np  # 引入 Numpy，用於數據處理

# 定義 DeepDream 模型
def deep_dream_model(layer_names):
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')  # 載入預訓練模型
    layers = [base_model.get_layer(name).output for name in layer_names]  # 提取指定層的輸出
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)  # 建立模型
    return dream_model

# 定義 DeepDream 處理
@tf.function
def deep_dream_step(img, model, step_size):
    with tf.GradientTape() as tape:  # 建立梯度計算的上下文
        tape.watch(img)  # 監視輸入影像
        outputs = model(img)  # 通過模型取得指定層的輸出
        loss = tf.reduce_sum([tf.reduce_mean(output) for output in outputs])  # 計算損失
    grads = tape.gradient(loss, img)  # 計算梯度
    grads /= tf.math.reduce_std(grads) + 1e-8  # 正規化梯度
    img += step_size * grads  # 更新影像
    img = tf.clip_by_value(img, 0.0, 1.0)  # 限制像素值範圍
    return loss, img

# 執行 DeepDream 特效
def apply_deep_dream(frame, model, step_size=0.01, dream_strength=1.0, steps=10):
    img = tf.convert_to_tensor(frame, dtype=tf.float32)  # 將影像轉為 TensorFlow 格式
    img = tf.expand_dims(img, axis=0)  # 添加 batch 維度
    for _ in range(steps):  # 多次迭代放大特效
        _, img = deep_dream_step(img, model, step_size * dream_strength)  # 應用特效
    img = tf.squeeze(img, axis=0)  # 移除 batch 維度
    return img.numpy()  # 返回處理後的影像

# Trackbar 回調函式（空函式，因為 OpenCV Trackbar 需要）
def nothing(x):
    pass

# 主程式
if __name__ == '__main__':
    dream_layers = ['mixed3', 'mixed5']  # 定義目標層
    dream_model = deep_dream_model(dream_layers)  # 建立 DeepDream 模型

    cap = cv2.VideoCapture(0)  # 開啟 Webcam（設備索引 0）
    if not cap.isOpened():  # 如果無法開啟攝影機
        print("無法開啟攝影機")
        exit()

    # 建立控制面板
    cv2.namedWindow("Controls")
    cv2.createTrackbar("Steps", "Controls", 10, 50, nothing)  # 用於控制迭代步數
    cv2.createTrackbar("Step Size", "Controls", 10, 100, nothing)  # 用於控制步伐大小
    cv2.createTrackbar("Strength", "Controls", 15, 50, nothing)  # 用於控制特效強度

    while True:
        ret, frame = cap.read()  # 捕捉影像
        if not ret:  # 如果捕捉失敗
            print("無法獲取影像")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將 BGR 格式轉為 RGB 格式
        frame = cv2.resize(frame, (224, 224))  # 調整影像大小以符合模型輸入
        frame = frame / 255.0  # 正規化影像到 0~1 範圍

        # 讀取 Trackbar 的值
        steps = cv2.getTrackbarPos("Steps", "Controls")  # 獲取步數
        step_size = cv2.getTrackbarPos("Step Size", "Controls") / 100.0  # 獲取步伐大小（0.01 ~ 1.0）
        dream_strength = cv2.getTrackbarPos("Strength", "Controls") / 10.0  # 獲取特效強度（0.1 ~ 5.0）

        # 應用 DeepDream 特效
        dream_frame = apply_deep_dream(frame, dream_model, step_size, dream_strength, steps)

        # 將影像轉換回 0~255 並顯示
        dream_frame = (dream_frame * 255).astype(np.uint8)  # 轉為 8-bit 格式
        dream_frame = cv2.cvtColor(dream_frame, cv2.COLOR_RGB2BGR)  # 將 RGB 格式轉回 BGR 格式

        # 顯示影像
        cv2.imshow("DeepDream Webcam", dream_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 釋放攝影機資源
    cv2.destroyAllWindows()  # 關閉所有視窗