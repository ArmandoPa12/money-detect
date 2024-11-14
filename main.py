from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = Flask(__name__)
# Directorio del proyecto
project_dir = os.path.dirname(os.path.abspath(__file__))

# Directorio de guardado del modelo
save_dir = os.path.join(project_dir, 'modelo')

# Cargar el modelo entrenado
model_path = os.path.join(save_dir, 'modelo1311.keras')
model = tf.keras.models.load_model(model_path)

# Diccionario de clases
class_indices = {0: '10', 1: '100', 2: '20', 3: '200', 4: '50'}


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/test")
def test():
    return jsonify({
        "message": "this is a test"
    })


@app.route("/predict", methods=["POST"])
def predict():
    # Verificar que se haya enviado un archivo
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Leer la imagen desde la solicitud y convertirla a un objeto BytesIO
    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert("RGB") 
    img = img.resize((224, 224))  # Ajustar el tamaño de la imagen a (224, 224)
    img_array = image.img_to_array(img) / 255.0  # Escalar los píxeles a [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para representar un solo lote

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_indices[predicted_class]
    confidence = float(predictions[0][predicted_class])

    predicted_class = np.argmax(predictions)
    predicted_label = class_indices[predicted_class]

    # Devolver el resultado en formato JSON
    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence*100
    })


@app.route("/predictv2", methods=["POST"])
def predictv2():
    # Verificar que se haya enviado un archivo
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Leer la imagen desde la solicitud y convertirla a un objeto BytesIO
    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert("RGB") 
    img = img.resize((224, 224))  # Ajustar el tamaño de la imagen a (224, 224)
    img_array = image.img_to_array(img) / 255.0  # Escalar los píxeles a [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para representar un solo lote

    confidence_threshold = float(request.form.get('confidence', 85)) / 100

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_indices[predicted_class]
    confidence = float(predictions[0][predicted_class])  # Mantener el nivel de confianza como un valor entre 0 y 1

    # Verificar si el nivel de confianza es suficiente
    if confidence < confidence_threshold:
        return jsonify({
            "predicted_label": "No se reconoce ningún billete",
            "confidence": f"{confidence * 100:.2f}%",
            "umbral":confidence_threshold
        })
    else:
        # Devolver la predicción con el nivel de confianza en porcentaje
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": f"{confidence * 100:.2f}%",
            "umbral":confidence_threshold
        })

@app.route("/inspect", methods=["POST"])
def inspect():
    # Verificar que se haya enviado un archivo
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Leer la imagen desde la solicitud y convertirla a un objeto BytesIO
    file = request.files['image']
    img = Image.open(BytesIO(file.read()))
    img_array = image.img_to_array(img)

    # Extraer dimensiones y formato de la imagen
    height, width, channels = img_array.shape
    format = file.content_type  # Obtiene el tipo MIME de la imagen

    # Devolver la información de la imagen en formato JSON
    return jsonify({
        "width": width,
        "height": height,
        "channels": channels,
        "format": format
    })