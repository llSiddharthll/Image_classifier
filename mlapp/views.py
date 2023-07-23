#mlapp/views.py
from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
from io import BytesIO
import json


# Load the pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

def preprocess_image(image):
    # Resize the image to the input size required by the model
    image = image.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Preprocess the image for the ResNet50 model
    processed_image = preprocess_input(image_array)

    # Expand the dimensions to create a batch of size 1 (required by the model)
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

def predict_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            # Get the uploaded image from the request
            uploaded_image = request.FILES['image']

            # Open the image using PIL
            image = Image.open(uploaded_image)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make predictions using the model
            predictions = model.predict(processed_image)

            # Decode the predictions
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            # Get the top predicted labels and probabilities
            top_predictions = [(label, float(f'{prob:.4f}')) for (_, label, prob) in decoded_predictions]

            # Return the predictions as JSON response
            return JsonResponse({'predictions': top_predictions})
        except Exception as e:
            # Handle errors here (optional)
            return JsonResponse({'error': 'An internal server error occurred'}, status=500)
    
    # Render the index.html template for GET requests
    return render(request, 'index.html')
