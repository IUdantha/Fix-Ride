from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# List of vehicle models
vehicle_model = [
    'Honda City 2000-2005', 'Nissan X-trail 2014-2016', 'Suzuki Swift 2000-2006',
    'Suzuki Swift 2017-2020', 'Suzuki Wagon R Stingray 2014-2016', 'Suzuki Wagon R Stingray 2017-2018',
    'Toyota Alion 240 2001-2007', 'Toyota Alion 260 2007-2010', 'Toyota Axio 2012-2015',
    'Toyota Axio 2015-2019', 'Toyota Premio 2007-2015'
]

# Dictionary for recommended parts for each vehicle model
recommended_parts = {
    'Honda City 2000-2005': ['Air Filter', 'Oil Filter', 'AC Filter'],
    'Nissan X-trail 2014-2016': ['Oil Filter', 'AC Filter', 'Air Filter'],
    'Suzuki Swift 2000-2006': ['Air Filter', 'AC Filter'],
    'Suzuki Swift 2017-2020': ['Oil Filter', 'Air Filter'],
    'Suzuki Wagon R Stingray 2014-2016': ['AC Filter', 'Oil Filter'],
    'Suzuki Wagon R Stingray 2017-2018': ['Air Filter', 'AC Filter'],
    'Toyota Alion 240 2001-2007': ['Oil Filter', 'AC Filter'],
    'Toyota Alion 260 2007-2010': ['Air Filter', 'Oil Filter'],
    'Toyota Axio 2012-2015': ['AC Filter', 'Oil Filter'],
    'Toyota Axio 2015-2019': ['Oil Filter', 'AC Filter'],
    'Toyota Premio 2007-2015': ['Air Filter', 'AC Filter', 'Oil Filter']
}

# Load the trained model
model = load_model('vehicle_model.h5')

# Function to classify the uploaded image
def classify_image(image) -> (str, float):
    img = Image.open(image)
    img = img.resize((256, 256))  # Resize image to match model input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict the class probabilities
    predictions = model.predict(img_array)

    # Get the predicted class and confidence score
    predicted_class = vehicle_model[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    
    return predicted_class, confidence

# Function to get recommended parts for the classified vehicle
def get_recommended_parts(vehicle: str):
    return recommended_parts.get(vehicle, [])

@app.post("/classify")
async def classify_vehicle(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected for uploading")

    try:
        # Classify the image
        image_data = await file.read()
        image = io.BytesIO(image_data)
        predicted_class, confidence = classify_image(image)
        parts = get_recommended_parts(predicted_class)
        
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'recommended_parts': parts
        }
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
