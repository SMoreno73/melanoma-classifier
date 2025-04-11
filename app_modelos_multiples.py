
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Clasificador de Melanoma", layout="centered")

st.title("🩺 Clasificador de Melanoma de Piel")

st.write("Sube una imagen de una lesión de piel para predecir si es **benigna** o **maligna** usando uno de los modelos disponibles.")

# Selector de modelo
modelo_opciones = {
    "Modelo basado en ceros": "modelo_ceros.h5",
    "Modelo Feature-based": "modelo_Feature-based.h5"
}
modelo_seleccionado = st.selectbox("Selecciona el modelo a usar", list(modelo_opciones.keys()))

@st.cache_resource
def cargar_modelo(modelo_path):
    return tf.keras.models.load_model(modelo_path)

# Cargar el modelo elegido
modelo = cargar_modelo(modelo_opciones[modelo_seleccionado])

# Subida de imagen
archivo_subido = st.file_uploader("📷 Sube una imagen de piel", type=["jpg", "jpeg", "png"])

if archivo_subido:
    imagen = Image.open(archivo_subido).resize((224, 224))
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    if st.button("🔍 Analizar imagen"):
        with st.spinner("Analizando..."):
            img_array = np.array(imagen) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = modelo.predict(img_array)[0][0]
            resultado = "🔴 Maligno" if pred > 0.5 else "🟢 Benigno"
            confianza = pred if pred > 0.5 else 1 - pred

            st.markdown(f"### Resultado: **{resultado}**")
            st.markdown(f"**Confianza:** {confianza:.2%}")
