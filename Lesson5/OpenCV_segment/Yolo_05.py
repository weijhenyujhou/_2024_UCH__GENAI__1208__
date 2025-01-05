import cv2
from ultralytics import YOLO

# 加載 YOLOv8 模型
# model = YOLO('yolov8n.pt')  # 確保權重檔案存在

# 加載 YOLOv11 模型
model = YOLO('yolo11n.pt')  # 確保權重檔案存在

# 啟用攝影機
cap = cv2.VideoCapture(1)  # 0 代表默認攝影機
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

# 讀取攝影機影像並進行檢測
while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機影像")
        break

    # YOLOv8 進行檢測
    results = model(frame)

    # 繪製檢測結果
    for result in results:
        for box in result.boxes:
            # 提取框座標、分類和信心分數
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()

            # 繪製矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 標註標籤和信心分數
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # 顯示檢測影像
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()