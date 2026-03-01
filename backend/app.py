from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load models
alzheimers_model = tf.keras.models.load_model('models/alzheimer_model.keras')
brainstroke_model = tf.keras.models.load_model('models/brainstroke_model.keras')
brain_tumor_model = tf.keras.models.load_model('models/brain_tumor_model.keras')

# Image preprocessing
def preprocess_image(image_stream):
    try:
        image = Image.open(image_stream).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return image_array.reshape(1, 224, 224, 3)
    except Exception as e:
        raise ValueError("Invalid image format or preprocessing error")

# Prediction logic
def predict_disease(model, image_array, disease_type):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    classes = {
        "Alzheimer's": ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'],
        "Brain Stroke": ['Haemorrhagic', 'Ischemic', 'Normal'],
        "Brain Tumor": ['Pituitary Tumor', 'No Tumor', 'Meningioma Tumor', 'Glioma Tumor']
    }

    if disease_type not in classes:
        raise ValueError("Invalid disease type")

    return classes[disease_type][predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        try:
            disease = request.form.get('disease')
            image_file = request.files.get('image')

            if not disease or not image_file:
                raise ValueError("Both disease and image must be provided")

            image_bytes = image_file.read()
            processed_image = preprocess_image(io.BytesIO(image_bytes))

            if disease == "Alzheimer's":
                result = predict_disease(alzheimers_model, processed_image, disease)
            elif disease == "Brain Stroke":
                result = predict_disease(brainstroke_model, processed_image, disease)
            elif disease == "Brain Tumor":
                result = predict_disease(brain_tumor_model, processed_image, disease)

        except Exception as e:
            error = str(e)

    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
