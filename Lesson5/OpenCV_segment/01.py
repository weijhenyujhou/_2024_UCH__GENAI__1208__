import cv2
import numpy as np

# 載入影像
image = cv2.imread('01.jpg')  # 讀取影像檔案
if image is None:  # 檢查是否成功載入影像
    raise FileNotFoundError("找不到 example.jpg，請確認影像路徑。")  # 若載入失敗則顯示錯誤訊息

# 設定影像寬度為 640，並計算等比高度
width = 640  # 目標寬度
height = int(image.shape[0] * (width / image.shape[1]))  # 根據寬高比例計算目標高度
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)  # 縮放影像到指定大小

# 創建遮罩
mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)  # 初始化遮罩，與影像大小相同，值為 0
rect = (10, 10, resized_image.shape[1] - 20, resized_image.shape[0] - 20)  # 定義矩形框範圍，前景應位於框內

# 初始化 GrabCut 所需的模型
bgd_model = np.zeros((1, 65), dtype=np.float64)  # 背景模型初始化
fgd_model = np.zeros((1, 65), dtype=np.float64)  # 前景模型初始化

# 執行 GrabCut 分割
cv2.grabCut(resized_image, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)  # 執行分割，迭代次數為 1

# 處理遮罩以區分前景和背景
mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')  # 將背景設為 0，前景設為 1
segmented_image = resized_image * mask2[:, :, np.newaxis]  # 將遮罩應用於影像，提取前景

# 顯示結果
cv2.imshow('Segmented Image', segmented_image)  # 顯示分割後的影像
cv2.waitKey(0)  # 等待按鍵輸入
cv2.destroyAllWindows()  # 關閉所有顯示視窗