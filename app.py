from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)



@app.route('/predict-kulit', methods=['POST'])
def predict():
    
    model = load_model("model.h5")

    # Kelas disesuaikan urutan saat training
    class_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Chickenpox', 'Dermato Fibroma',
                'Dyshidrotic Eczema', 'Melanoma', 'Nail Fungus', 'Nevus', 'Normal Skin',
                'Pigmented Benign Keratosis', 'Ringworm', 'Seborrheic Keratosis',
                'Squamous Cell Carcinoma', 'Vascular Lesion']

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        img = Image.open(file.stream).resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        #melakukan prediksi
        pred = model.predict(img_array)
        class_index = np.argmax(pred[0])
        class_name = class_names[class_index]
        confidence = float(np.max(pred[0]))

        return jsonify({'class': class_name, 'confidence': round(confidence, 3)})
    except KeyError:
        return jsonify({ "status": "fail", "error": KeyError}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)