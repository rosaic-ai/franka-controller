import cv2

device_index = 6  # USB 카메라의 인덱스 (위에서 확인한 값)

cap = cv2.VideoCapture(device_index)
if not cap.isOpened():
    print("Camera could not be opened.")
else:
    print("Camera opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        cv2.imshow("USB Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()