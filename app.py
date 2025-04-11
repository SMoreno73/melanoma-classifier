
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Clasificador de Melanoma", layout="centered")
st.title("🩺 Clasificador de Melanoma de Piel")

st.write("Sube una imagen de una lesión de piel para predecir si es **benigna** o **maligna** usando uno de los modelos disponibles.")

# URLs de los modelos en Google Drive
model_urls = {
    "Modelo basado en ceros": "https://drive.google.com/uc?id=1WO-3RucWsrWW_6HQeaPE3cq-UFUwPGCa",
    "Modelo Feature-based": "https://drive.google.com/uc?id=1cbAS5yZcWMxYVpZRf1Whv7jJENdu1Iqe"
}

# Archivos locales
model_files = {
    "Modelo basado en ceros": "modelo_ceros.h5",
    "Modelo Feature-based": "modelo_Feature-based.h5"
}

# Selección de modelo
modelo_seleccionado = st.selectbox("Selecciona el modelo a usar", list(model_urls.keys()))
modelo_path = model_files[modelo_seleccionado]

# Descargar si no existe
if not os.path.exists(modelo_path):
    with st.spinner(f"Descargando {modelo_seleccionado}..."):
        gdown.download(model_urls[modelo_seleccionado], modelo_path, quiet=False)

@st.cache_resource
def cargar_modelo(ruta):
    return tf.keras.models.load_model(ruta)

modelo = cargar_modelo(modelo_path)

# Subida de imagen
archivo_subido = st.file_uploader("📷 Sube una imagen de piel", type=["jpg", "jpeg", "png"])

if archivo_subido:
    imagen = Image.open(archivo_subido)
    input_shape = modelo.input_shape[1:]  # omitir el batch size

    try:
        if input_shape == (224, 224, 3) or len(input_shape) == 3:
            imagen = imagen.resize((input_shape[1], input_shape[0]))
            img_array = np.array(imagen) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        elif len(input_shape) == 1:
            # Imagen aplanada
            imagen = imagen.resize((224, 224))  # asumimos que es 224*224*3
            img_array = np.array(imagen).reshape(1, -1) / 255.0
        else:
            st.error(f"⚠️ No se puede interpretar la forma esperada: {input_shape}")
            st.stop()

        st.image(imagen, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Analizar imagen"):
            with st.spinner("Analizando..."):
                pred = modelo.predict(img_array)[0][0]
                resultado = "🔴 Maligno" if pred > 0.5 else "🟢 Benigno"
                confianza = pred if pred > 0.5 else 1 - pred

                st.markdown(f"### Resultado: **{resultado}**")
                st.markdown(f"**Confianza:** {confianza:.2%}")
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
