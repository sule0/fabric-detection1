from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.api.models import load_model
import tensorflow as tf
from keras.src.legacy.preprocessing import image
import numpy as np
import io
import os

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)  # Tüm originlere izin vermek için

# Modeli yükle
model = load_model('kumas_tipi_model.h5')

# Sınıf etiketlerini tanımla
class_labels = ['stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']  # Kendi sınıf etiketlerinizi buraya ekleyin

# Resim boyutları
img_height = 224
img_width = 224

# Geçici depolama
training_data = []
training_labels = []
training_threshold = 10  # Eğitim eşiği

# Ana sayfa
@app.route('/')
def index():
    return "Kumaş Tipi Tanıma API'sine Hoş Geldiniz"

# Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        img = image.image_utils.load_img(io.BytesIO(file.read()), target_size=(img_height, img_width))
        img_array = image.image_utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Resmi 4 boyutlu hale getirir (1, img_height, img_width, 3)
        img_array /= 255.0  # Normalizasyon

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = predictions[0][predicted_class_index] * 100

        return jsonify({'predicted_class': class_labels[predicted_class[0]], 'confidence_score': f'{confidence_score:.2f}%'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Modeli güncelleme endpoint'i
@app.route('/update', methods=['POST'])
def update_model():
    global training_data, training_labels

    data = request.json
    if 'image' not in data or 'correct_class' not in data:
        return jsonify({'error': 'Invalid request data'})
    
    image_uri = data['image']
    correct_class = data['correct_class']
    
    try:
        # Resmi yükle ve işleme
        img = image.image_utils.load_img(image_uri, target_size=(img_height, img_width))
        img_array = image.image_utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Doğru sınıfın indeksini bul
        correct_class_index = class_labels.index(correct_class)
        
        # Geçici depolama
        training_data.append(img_array)
        training_labels.append(correct_class_index)
        
        # Eğitim eşiğine ulaşıldığında modeli yeniden eğitme
        if len(training_data) >= training_threshold:
            training_data_np = np.vstack(training_data)
            training_labels_np = np.array(training_labels)
            
            # Modeli yeniden eğitme
            model.fit(training_data_np, training_labels_np, epochs=5, verbose=1)
            
            # Modeli kaydet
            model.save('kumas_tipi_model_updated.h5')
            
            # Geçici depolamayı temizle
            training_data = []
            training_labels = []
        
        return jsonify({'message': 'Model güncelleme için veri alındı'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
