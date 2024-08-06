import streamlit as st
import requests
from PIL import Image
import io

st.title("Fabric Type Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # Görüntüyü yükleyin ve gösterin
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # API'ye gönderilecek dosya
    files = {'file': uploaded_file.getvalue()}

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            try:
                # API'ye isteği gönderin
                response = requests.post("http://localhost/predict", files=files)
                response.raise_for_status()  # Hataları kontrol et

                # Yanıtı işleyin
                result = response.json()
                st.success(f'Predicted Class: {result["predicted_class"]}')
            except requests.exceptions.RequestException as e:
                st.error(f'Error: {e}')
