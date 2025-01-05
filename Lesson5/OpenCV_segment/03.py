import cv2
import numpy as np

# 回調函數，用於更新 trackbar


def update_display(val):
    global display_image, sorted_contours, resized_image
    num_objects = cv2.getTrackbarPos('Objects', 'Detected Objects')
    display_image = resized_image.copy()

    # 繪製框，僅保留面積最大的指定數量
    for i, contour in enumerate(sorted_contours[:num_objects]):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(display_image, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # 畫出綠色框

    cv2.imshow('Detected Objects', display_image)


# 載入影像
image = cv2.imread('01.jpg')
if image is None:
    raise FileNotFoundError("找不到 01.jpg，請確認影像路徑。")

# 調整影像寬度至 640，保持比例
scale = 640 / image.shape[1]
new_dimensions = (640, int(image.shape[0] * scale))
resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

# 將影像轉為灰階
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 應用二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 找出影像中的輪廓
contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按面積由大到小排序輪廓
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 初始化視窗和 Trackbar
cv2.namedWindow('Detected Objects')
cv2.createTrackbar('Objects', 'Detected Objects', 1, 20, update_display)

# 顯示初始影像
display_image = resized_image.copy()
update_display(0)

# 等待使用者操作
cv2.waitKey(0)
cv2.destroyAllWindows()
