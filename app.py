from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.src import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout ,InputLayer
from keras.api.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing import image
import numpy as np
import io
import os
from werkzeug.utils import secure_filename

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)  # Tüm originlere izin vermek için

# Modeli yükle
model = load_model('kumas_hasar_model.h5')

# Sınıf etiketlerini tanımla
class_labels = ['hole', 'horizontal', 'verticle']  # Kendi sınıf etiketlerinizi buraya ekleyin

# Resim boyutları
img_height = 224
img_width = 224

# Eğitim verileri için dizin
data_dir = 'dataset'  # Eğitim verilerinin dizini
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ana sayfa
@app.route('/')
def index():
    return "Kumaş Hasar Tespiti API'sine Hoş Geldiniz"

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
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = predictions[0][predicted_class_index] * 100

        return jsonify({'predicted_class': class_labels[predicted_class_index], 'confidence_score': f'{confidence_score:.2f}%'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Yeniden eğitme endpoint'i
@app.route('/retrain', methods=['POST'])
def retrain():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Fotoğrafı kaydet
    filename = secure_filename(file.filename)
    file_path = os.path.join(data_dir, filename)
    file.save(file_path)

    # Eğitim ve doğrulama verilerini oluştur
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Normalizasyon
        validation_split=0.2,       # Eğitim ve doğrulama setlerine ayrım
        rotation_range=40,          # Resimleri döndürme
        width_shift_range=0.2,      # Genişlik kaydırma
        height_shift_range=0.2,     # Yükseklik kaydırma
        shear_range=0.2,            # Kesme kaydırması
        zoom_range=0.2,             # Yakınlaştırma
        horizontal_flip=True,       # Yatay çevirme
        fill_mode='nearest'         # Doldurma modu
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Modeli yeniden tanımla
    model = Sequential([
        InputLayer(input_shape=(img_height, img_width, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(class_labels), activation='softmax')  # Dinamik sınıf sayısı
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli yeniden eğit
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator
    )

    # Yeni modeli kaydet
    model.save('kumas_hasar_model.h5')

    return jsonify({'message': 'Model başarıyla yeniden eğitildi ve kaydedildi'})

if __name__ == '__main__':
    app.run(debug=True)
