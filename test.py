import numpy as np
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO

#############################################
# CAMERA SETTINGS
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75  # Probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# LOAD YOLOv12 MODEL FOR DETECTION
yolo_model = YOLO("yolo12n.pt")  # หรือใช้โมเดล YOLO ที่เทรนกับป้ายจราจร
# LOAD CNN MODEL FOR CLASSIFICATION
cnn_model = load_model("model.h5")

def preprocess_image(img):
    """
    แปลงภาพเป็น grayscale, normalize, และ resize ให้ CNN ใช้
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0  # Normalize
    img = cv2.resize(img, (32, 32))  # Resize ให้ CNN ใช้
    img = img.reshape(1, 32, 32, 1)  # Reshape เป็น input ของ CNN
    return img

def getClassName(classNo):
    """
    คืนค่าชื่อของป้ายจราจร
    """
    class_labels = {
            0: ("Speed Limit 20 km/h", "จำกัดความเร็ว 20 กม./ชม."),
            1: ("Speed Limit 30 km/h", "จำกัดความเร็ว 30 กม./ชม."),
            2: ("Speed Limit 50 km/h", "จำกัดความเร็ว 50 กม./ชม."),
            3: ("Speed Limit 60 km/h", "จำกัดความเร็ว 60 กม./ชม."),
            4: ("Speed Limit 70 km/h", "จำกัดความเร็ว 70 กม./ชม."),
            5: ("Speed Limit 80 km/h", "จำกัดความเร็ว 80 กม./ชม."),
            6: ("End of Speed Limit 80 km/h", "สิ้นสุดการจำกัดความเร็ว 80 กม./ชม."),
            7: ("Speed Limit 100 km/h", "จำกัดความเร็ว 100 กม./ชม."),
            8: ("Speed Limit 120 km/h", "จำกัดความเร็ว 120 กม./ชม."),
            9: ("No Passing", "ห้ามแซง"),
            10: ("No Passing for Vehicles Over 3.5 Metric Tons", "ห้ามแซงสำหรับยานพาหนะที่มีน้ำหนักเกิน 3.5 ตัน"),
            11: ("Right-of-Way at Next Intersection", "ให้สิทธิ์ทางที่ทางแยกข้างหน้า"),
            12: ("Priority Road", "ให้รถสวนทางมาก่อน"),
            13: ("Yield", "ให้ทาง"),
            14: ("Stop", "หยุด"),
            15: ("No Vehicles", "ห้ามยานพาหนะ"),
            16: ("Vehicles Over 3.5 Metric Tons Prohibited", "ห้ามยานพาหนะที่มีน้ำหนักเกิน 3.5 ตัน"),
            17: ("No Entry", "ห้ามเข้า"),
            18: ("General Caution", "ระวังอันตราย"),
            19: ("Dangerous Curve to the Left", "โค้งอันตรายไปทางซ้าย"),
            20: ("Dangerous Curve to the Right", "โค้งอันตรายไปทางขวา"),
            21: ("Double Curve", "โค้งต่อเนื่อง"),
            22: ("Bumpy Road", "ถนนขรุขระ"),
            23: ("Slippery Road", "ถนนลื่น"),
            24: ("Road Narrows on the Right", "ถนนแคบลงทางขวา"),
            25: ("Road Work", "มีการก่อสร้างถนน"),
            26: ("Traffic Signals", "สัญญาณไฟจราจร"),
            27: ("Pedestrians", "คนเดินเท้า"),
            28: ("Children Crossing", "ทางข้ามเด็กนักเรียน"),
            29: ("Bicycles Crossing", "ทางข้ามจักรยาน"),
            30: ("Beware of Ice/Snow", "ระวังน้ำแข็ง/หิมะ"),
            31: ("Wild Animals Crossing", "ทางข้ามสัตว์ป่า"),
            32: ("End of All Speed and Passing Limits", "สิ้นสุดข้อจำกัดความเร็วและการแซง"),
            33: ("Turn Right Ahead", "เลี้ยวขวาข้างหน้า"),
            34: ("Turn Left Ahead", "เลี้ยวซ้ายข้างหน้า"),
            35: ("Ahead Only", "ตรงไปเท่านั้น"),
            36: ("Go Straight or Right", "ตรงไปหรือเลี้ยวขวา"),
            37: ("Go Straight or Left", "ตรงไปหรือเลี้ยวซ้าย"),
            38: ("Keep Right", "ชิดขวา"),
            39: ("Keep Left", "ชิดซ้าย"),
            40: ("Roundabout Mandatory", "ต้องใช้วงเวียน"),
            41: ("End of No Passing", "สิ้นสุดข้อห้ามแซง"),
            42: ("End of No Passing by Vehicles Over 3.5 Metric Tons", "สิ้นสุดข้อห้ามแซงสำหรับยานพาหนะที่มีน้ำหนักเกิน 3.5 ตัน")
        }
    return class_labels.get(classNo, "Unknown Sign")

while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    # YOLO DETECTION
    results = yolo_model(imgOriginal)
    detected = False

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding Box

            # ครอบเฉพาะป้ายจราจร
            cropped_sign = imgOriginal[y1:y2, x1:x2]

            # ส่งภาพให้ CNN จำแนก
            imgCNN = preprocess_image(cropped_sign)
            predictions = cnn_model.predict(imgCNN)
            classIndex = np.argmax(predictions, axis=1)[0]  # Class ที่มีค่าสูงสุด
            probabilityValue = np.max(predictions)  # ความมั่นใจของการพยากรณ์

            # วาดกรอบสี่เหลี่ยม และแสดงชื่อป้าย
            label = f"{getClassName(classIndex)} ({round(probabilityValue * 100, 2)}%)"
            cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(imgOriginal, label, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)

            detected = True

    # ถ้าไม่พบป้ายจราจร แสดงข้อความ
    if not detected:
        cv2.putText(imgOriginal, "No Traffic Sign Detected", (20, 50), font, 0.75, (0, 0, 255), 2)

    cv2.imshow("Traffic Sign Detection", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
