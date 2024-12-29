import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Replace 0 with the detected index if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print(f"Backend: {cap.getBackendName()}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
