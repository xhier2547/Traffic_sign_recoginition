# 🚦 Traffic Sign Recognition

**Traffic Sign Recognition** เป็นโปรเจค Deep Learning ที่ใช้ **Convolutional Neural Network (CNN)** ในการจำแนกป้ายจราจรจากภาพถ่าย  
พัฒนาโดยใช้ **TensorFlow, Keras และ OpenCV** พร้อมรองรับการใช้งานผ่าน **Flask Web Application**  

---

## 📌 คุณสมบัติหลัก
- ✅ ใช้ CNN ในการตรวจจับและจำแนกป้ายจราจร
- ✅ รองรับ GPU เพื่อเพิ่มความเร็วในการ Train โมเดล(สำหรับคนที่การ์ดจอรุ่นใหม่ๆ หรือคนที่ใช้การ์ดจอ nvidia)
- ✅ ใช้ **Data Augmentation** เพื่อลด Overfitting
- ✅ ใช้ **Flask Web App** สำหรับทดสอบโมเดลแบบ UI
- ✅ มีระบบ **Virtual Environment (venv/.venv)** เพื่อแยก dependencies
- ✅ สามารถดาวน์โหลดและเพิ่ม Dataset ได้เอง

---

## วิธีติดตั้งและใช้งาน

### **1. Clone โปรเจค**
```bash
git clone https://github.com/xhier2547/Traffic_sign_recoginition.git
cd Traffic_sign_recoginition
```

### **2.  สร้างและเปิดใช้งาน Virtual Environment**
🔹 Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

🔹 macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **3. ติดตั้ง Dependencies(requirements)**
```bash
pip install -r requirements.txt
```

### **4.  ดาวน์โหลด Dataset**

https://drive.google.com/drive/folders/1dlVGudBMMXZLF2dJeVA1V7j6ybajvXIV?usp=sharing


*หลังจากนั้นให้ทำการแตกไฟล์ไว้ในโฟลเดอร์ของโปรเจค โดยมีชื่อโฟลเดอร์ว่า Dataset และ ภายในประกอบด้วย class ทั้ง 44(0-43) class*

### **5.  การ Train โมเดล**

เปิดไฟล์ Traffic_Sign_Recognition.ipynb แล้วรัน

### **6️. การรัน Flask Web App**
*1. เปิด Virtual Environment ก่อน (ถ้ายังไม่ได้เปิด)*
```bash
# บน Windows
.venv\Scripts\activate

# บน macOS/Linux
source .venv/bin/activate

```

*2. รัน Flask Server, app.py*
```bash
python app.py

```

*3. เปิด **http://127.0.0.1:5000** ในเว็บเบราว์เซอร์*


---

### **หมายเหตุ**
**ไฟล์ model.h5 เป็นไฟล์โมเดลที่ Train แล้ว สามารถใช้ทดสอบได้ทันที**
**หากต้องการ Train โมเดลใหม่ ควรมี Dataset ป้ายจราจร พร้อมใช้งาน**

---

### **ติดต่อ**
**Discord : xhao.none**

**Email : aphilakjankaewdach@gmail.com**