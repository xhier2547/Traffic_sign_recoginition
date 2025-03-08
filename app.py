from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)
model = load_model("model.h5")  # โหลดโมเดล CNN ที่เทรนไว้
yolo_model = YOLO("yolo12n.pt")  # โหลดโมเดล YOLOv12

def detect_traffic_sign_yolo12(image_path, output_path="static/detected_image.jpg"):
    """
    ใช้ YOLOv12 เพื่อตรวจจับป้ายจราจร และครอบเฉพาะป้าย
    """
    img = cv2.imread(image_path)
    results = yolo_model(image_path)  # ตรวจจับวัตถุด้วย YOLOv12
    detected = False
    cropped_sign = None  

    for result in results:
        for box in result.boxes.xyxy:  # วนลูปอ่าน bounding box
            x1, y1, x2, y2 = map(int, box[:4])  # พิกัดกรอบป้ายจราจร
            detected = True

            # วาดกรอบสี่เหลี่ยม
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Traffic Sign", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ครอบเฉพาะป้ายจราจร
            cropped_sign = img[y1:y2, x1:x2]

            # บันทึกภาพที่มี Bounding Box
            cv2.imwrite(output_path, img)
            return cropped_sign, output_path  # คืนค่า (ภาพป้าย, ภาพที่มี bounding box)

    # ถ้าไม่มีการตรวจจับ ให้ใช้ภาพต้นฉบับ
    cv2.imwrite(output_path, img)
    return img, output_path  # คืนค่า (ภาพต้นฉบับ, ภาพที่ไม่มีการ detect)

def preprocess_and_predict(image_path):
    """
    ใช้ YOLO ตรวจจับป้ายจราจรก่อน แล้วทำการ preprocessing ให้ CNN ใช้งาน
    จากนั้นนำไป predict ด้วย model.h5
    """
    cropped_sign, detected_img_path = detect_traffic_sign_yolo12(image_path)

    if cropped_sign is None:
        return "Unknown Sign", "ป้ายไม่รู้จัก", detected_img_path  # ถ้าไม่เจอป้าย ให้คืนค่าป้ายไม่รู้จัก

    # แปลงเป็น grayscale และทำ preprocessing
    img = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0  # Normalize
    img = cv2.resize(img, (32,32))  # Resize ให้ CNN ใช้
    img = img.reshape(1, 32, 32, 1)  # ปรับเป็นรูปแบบที่ CNN ต้องการ

    # Predict กับ model.h5
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # เลือก class ที่มีค่ามากสุด

    # Mapping Class Number to Sign Name
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
    predicted_label = class_labels.get(predicted_class, ("Unknown Sign", "ป้ายไม่รู้จัก"))
    return predicted_label[0], predicted_label[1], detected_img_path  # (อังกฤษ, ไทย, ภาพที่มี bounding box)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_en = None
    prediction_th = None
    image_path = None
    detected_image_path = None

    if request.method == "POST":
        file = request.files["image"]
        file_path = "static/uploaded_image.jpg"
        file.save(file_path)  # บันทึกไฟล์ภาพ

        # ตรวจจับป้ายจราจร + Predict ด้วย CNN
        prediction_en, prediction_th, detected_image_path = preprocess_and_predict(file_path)

        return render_template("index.html", prediction_en=prediction_en, prediction_th=prediction_th, 
                               image=file_path, detected_image=detected_image_path)

    return render_template("index.html", prediction=prediction_en)

if __name__ == "__main__":
    app.run(debug=True)
