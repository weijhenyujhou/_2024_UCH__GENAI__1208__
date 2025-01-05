import cv2
import numpy as np

# 定義回調函數（Trackbar 的佔位函數）
def nothing(x):
    pass  # 不執行任何動作

# 開啟 Webcam
cap = cv2.VideoCapture(0)  # 使用 Webcam（通常為攝像頭 0）
if not cap.isOpened():
    raise Exception("無法開啟攝像頭")  # 檢查攝像頭是否成功啟動

# 創建調整參數的視窗和 Trackbar
cv2.namedWindow('Segmented Image')  # 創建顯示視窗
cv2.createTrackbar('Iterations', 'Segmented Image', 1, 10, nothing)  # 設定迭代次數的 Trackbar
cv2.createTrackbar('Rect Size', 'Segmented Image', 50, 200, nothing)  # 設定矩形框大小的 Trackbar

while True:
    ret, frame = cap.read()  # 從攝像頭讀取一幀影像
    if not ret:
        print("無法讀取影像")
        break

    # 獲取當前 Trackbar 的值
    iterations = cv2.getTrackbarPos('Iterations', 'Segmented Image')  # 獲取迭代次數
    rect_size = cv2.getTrackbarPos('Rect Size', 'Segmented Image')  # 獲取矩形框大小

    # 縮放影像到固定寬度 640，等比調整高度
    width = 320  # 固定寬度
    height = int(frame.shape[0] * (width / frame.shape[1]))  # 計算高度
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)  # 縮放影像

    # 確保矩形框不超出影像範圍
    rect_x = max(0, rect_size)  # 確保 X 起點不小於 0
    rect_y = max(0, rect_size)  # 確保 Y 起點不小於 0
    rect_width = min(resized_frame.shape[1] - rect_x, resized_frame.shape[1] - rect_size * 2)  # 限制寬度
    rect_height = min(resized_frame.shape[0] - rect_y, resized_frame.shape[0] - rect_size * 2)  # 限制高度

    if rect_width <= 0 or rect_height <= 0:  # 檢查矩形框是否合法
        print("矩形框大小無效，請調整 Rect Size")
        continue

    rect = (rect_x, rect_y, rect_width, rect_height)  # 定義合法的矩形框範圍

    # 初始化遮罩與模型
    mask = np.zeros(resized_frame.shape[:2], dtype=np.uint8)  # 創建空白遮罩
    bgd_model = np.zeros((1, 65), dtype=np.float64)  # 背景模型
    fgd_model = np.zeros((1, 65), dtype=np.float64)  # 前景模型

    # 執行 GrabCut 分割
    try:
        cv2.grabCut(resized_frame, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)  # GrabCut 分割
    except Exception as e:
        print(f"GrabCut 執行錯誤: {e}")
        continue

    # 處理遮罩
    mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')  # 背景設為 0，前景設為 1
    segmented_image = resized_frame * mask2[:, :, np.newaxis]  # 使用遮罩提取前景

    # 顯示結果
    cv2.imshow('Original', resized_frame)  # 顯示原始影像
    cv2.imshow('Segmented Image', segmented_image)  # 顯示分割後影像

    # 按下 'q' 鍵退出程式
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉視窗
cap.release()  # 關閉攝

"""
詳解
cv2.GC_BGD (0):

明確標記為背景，完全不屬於前景。
cv2.GC_FGD (1):

明確標記為前景，完全屬於感興趣的物件或區域。
cv2.GC_PR_BGD (2):

可能是背景，但不確定，GrabCut 會嘗試在進一步的迭代中重新分類這些像素。
cv2.GC_PR_FGD (3):

可能是前景，但不確定，GrabCut 會嘗試在進一步的迭代中重新分類這些像素。
"""

