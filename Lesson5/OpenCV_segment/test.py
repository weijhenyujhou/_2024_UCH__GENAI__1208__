import cv2
import numpy as np

# 創建一個簡單的圖片
img = np.zeros((100, 100, 3), dtype=np.uint8)
# 儲存為檔案（不需要顯示視窗）
cv2.imwrite('test.jpg', img)

