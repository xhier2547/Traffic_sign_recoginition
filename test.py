import numpy as np
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# CAMERA SETTINGS
frameWidth = 640
frameHeight = 480
brightness = 180
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# LOAD YOLO MODEL
yolo_model = YOLO("yolo12n.pt")  # ใช้ YOLO ตรวจจับป้ายจราจร
cnn_model = load_model("model.h5")  # โหลด CNN สำหรับจำแนกประเภทป้าย

# YOLO Class Labels
yolo_classes = yolo_model.names

# **คลาสของป้ายจราจร 43 ประเภท**
traffic_sign_labels = {
    0: "Speed Limit 20 km/h", 1: "Speed Limit 30 km/h", 2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h", 4: "Speed Limit 70 km/h", 5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h", 7: "Speed Limit 100 km/h", 8: "Speed Limit 120 km/h",
    9: "No Passing", 10: "No Passing for Vehicles Over 3.5 Metric Tons",
    11: "Right-of-Way at Next Intersection", 12: "Priority Road",
    13: "Yield", 14: "Stop", 15: "No Vehicles",
    16: "Vehicles Over 3.5 Metric Tons Prohibited", 17: "No Entry",
    18: "General Caution", 19: "Dangerous Curve to the Left",
    20: "Dangerous Curve to the Right", 21: "Double Curve",
    22: "Bumpy Road", 23: "Slippery Road",
    24: "Road Narrows on the Right", 25: "Road Work",
    26: "Traffic Signals", 27: "Pedestrians",
    28: "Children Crossing", 29: "Bicycles Crossing",
    30: "Beware of Ice/Snow", 31: "Wild Animals Crossing",
    32: "End of All Speed and Passing Limits", 33: "Turn Right Ahead",
    34: "Turn Left Ahead", 35: "Ahead Only",
    36: "Go Straight or Right", 37: "Go Straight or Left",
    38: "Keep Right", 39: "Keep Left",
    40: "Roundabout Mandatory", 41: "End of No Passing",
    42: "End of No Passing by Vehicles Over 3.5 Metric Tons"
}

def preprocess_image(img):
    """แปลงภาพเป็น grayscale, normalize, และ resize ให้ CNN ใช้"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 1)
    return img

while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    results = yolo_model(imgOriginal)
    detected = False

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = yolo_classes[int(cls)]  # อ่าน Label ของ YOLO
            print(f"YOLO Detected: {class_name}")  # Debug

            # **ใช้ CNN จำแนกป้ายจราจร**
            cropped_sign = imgOriginal[y1:y2, x1:x2]

            # เช็คว่า cropped_sign มีขนาดเพียงพอไหม
            if cropped_sign.shape[0] == 0 or cropped_sign.shape[1] == 0:
                continue  # ถ้าขนาดไม่ถูกต้อง ให้ข้ามไป

            imgCNN = preprocess_image(cropped_sign)
            predictions = cnn_model.predict(imgCNN)
            classIndex = np.argmax(predictions, axis=1)[0]  # Class ที่มีค่าสูงสุด
            probabilityValue = np.max(predictions)  # ความมั่นใจของการพยากรณ์

            # **ดึงชื่อป้ายจราจรจาก ClassIndex**
            label = f"{traffic_sign_labels.get(classIndex, 'Unknown Sign')} ({round(probabilityValue * 100, 2)}%)"
            cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(imgOriginal, label, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)

            detected = True

    if not detected:
        cv2.putText(imgOriginal, "No Traffic Sign Detected", (20, 50), font, 0.75, (0, 0, 255), 2)

    cv2.imshow("Traffic Sign Detection", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
