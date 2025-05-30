﻿# 🩺 Clasificador de Melanoma con Deep Learning

Esta aplicación web permite clasificar imágenes de lesiones en la piel como **benignas** o **malignas**, usando modelos de aprendizaje profundo entrenados con TensorFlow.

## 🚀 Demo en línea

Accede a la app desplegada aquí:  
👉 [[https://nombreusuario.streamlit.app](https://kzy9sslhk9cpot5zw6gtar.streamlit.app/)]([https://nombreusuario.streamlit.app](https://kzy9sslhk9cpot5zw6gtar.streamlit.app/))

## 📂 Modelos disponibles

Puedes elegir entre dos modelos entrenados:
- `modelo_ceros.h5`: basado en características estadísticas.
- `modelo_Feature-based.h5`: basado en ingeniería de características.

## 🖼 ¿Cómo usarla?

1. Selecciona el modelo que deseas usar en el menú desplegable.
2. Sube una imagen de una lesión cutánea (`.jpg`, `.jpeg`, `.png`).
3. Haz clic en **"🔍 Analizar imagen"**.
4. La app mostrará si la lesión es **benigna** o **maligna**, junto con la probabilidad estimada.

## 🛠 Requisitos

Este proyecto necesita las siguientes dependencias, que están en `requirements.txt`:

- `streamlit`
- `tensorflow`
- `pillow`
- `numpy`

## ⚙️ Ejecución local

```bash
git clone https://github.com/tu_usuario/melanoma-classifier.git
cd melanoma-classifier
pip install -r requirements.txt
streamlit run app_modelos_multiples.py
