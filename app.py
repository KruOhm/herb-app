from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO
import time

# สร้าง Flask app
app = Flask(__name__)

# โหลดโมเดลที่ฝึกไว้
model = load_model("herb_classifier_mobilenetv2.h5")

# ชื่อ class ของสมุนไพร (ตามที่ train)
class_names = ['coffee', 'kraprao', 'noni', 'papaya', 'saabseua']  # ปรับให้ตรง dataset

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            # กรณีอัปโหลดไฟล์
            file = request.files['image']
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
        elif 'camera_image' in request.form and request.form['camera_image'] != '':
            # กรณีถ่ายจากกล้อง (base64)
            image_data = request.form['camera_image'].split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))
            img_path = f'static/captured_{int(time.time())}.jpg'
            img.save(img_path)
        else:
            return render_template('index.html', prediction="ไม่พบรูปภาพ")

        # เตรียมภาพสำหรับโมเดล
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ทำนายผล
        pred = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(pred)]
        prediction = f"{predicted_class} ({np.max(pred)*100:.2f}%)"

        return render_template('index.html', prediction=prediction, image_url=img_path)

    return render_template('index.html')

# ✅ ส่วนสำคัญที่สุด เพื่อให้ Flask รันได้
if __name__ == '__main__':
    app.run(debug=True)
