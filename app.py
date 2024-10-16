from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# List of vehicle models
vehicle_model = [
    'Honda City 2000-2005', 'Nissan X-trail 2014-2016', 'Suzuki Swift 2000-2006',
    'Suzuki Swift 2017-2020', 'Suzuki Wagon R Stingray 2014-2016', 'Suzuki Wagon R Stingray 2017-2018',
    'Toyota Alion 240 2001-2007', 'Toyota Alion 260 2007-2010', 'Toyota Axio 2012-2015',
    'Toyota Axio 2015-2019', 'Toyota Premio 2007-2015'
]

# Extended dictionary for recommended parts with vendor-specific prices for all vehicles
recommended_parts = {
    'Honda City 2000-2005': [
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor A', 'price': 15.00}, {'name': 'Vendor B', 'price': 16.00}]},
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor C', 'price': 10.00}, {'name': 'Vendor D', 'price': 11.00}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor E', 'price': 12.00}, {'name': 'Vendor F', 'price': 13.00}]}
    ],
    'Nissan X-trail 2014-2016': [
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor A', 'price': 20.00}, {'name': 'Vendor G', 'price': 19.50}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor B', 'price': 15.00}, {'name': 'Vendor H', 'price': 14.75}]},
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor C', 'price': 18.00}, {'name': 'Vendor I', 'price': 17.50}]}
    ],
    'Suzuki Swift 2000-2006': [
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor J', 'price': 14.00}, {'name': 'Vendor K', 'price': 15.50}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor L', 'price': 13.00}, {'name': 'Vendor M', 'price': 13.50}]}
    ],
    'Suzuki Swift 2017-2020': [
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor N', 'price': 12.00}, {'name': 'Vendor O', 'price': 12.50}]},
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor P', 'price': 14.50}, {'name': 'Vendor Q', 'price': 15.00}]}
    ],
    'Suzuki Wagon R Stingray 2014-2016': [
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor R', 'price': 10.00}, {'name': 'Vendor S', 'price': 10.50}]},
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor T', 'price': 11.50}, {'name': 'Vendor U', 'price': 12.00}]}
    ],
    'Suzuki Wagon R Stingray 2017-2018': [
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor V', 'price': 15.00}, {'name': 'Vendor W', 'price': 16.00}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor X', 'price': 13.00}, {'name': 'Vendor Y', 'price': 13.50}]}
    ],
    'Toyota Alion 240 2001-2007': [
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor Z', 'price': 18.00}, {'name': 'Vendor A1', 'price': 18.50}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor B1', 'price': 16.00}, {'name': 'Vendor C1', 'price': 16.50}]}
    ],
    'Toyota Alion 260 2007-2010': [
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor D1', 'price': 19.00}, {'name': 'Vendor E1', 'price': 19.50}]},
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor F1', 'price': 17.00}, {'name': 'Vendor G1', 'price': 17.50}]}
    ],
    'Toyota Axio 2012-2015': [
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor H1', 'price': 15.50}, {'name': 'Vendor I1', 'price': 16.00}]},
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor J1', 'price': 18.00}, {'name': 'Vendor K1', 'price': 18.50}]}
    ],
    'Toyota Axio 2015-2019': [
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor L1', 'price': 19.00}, {'name': 'Vendor M1', 'price': 19.50}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor N1', 'price': 14.00}, {'name': 'Vendor O1', 'price': 14.50}]}
    ],
    'Toyota Premio 2007-2015': [
        {'part': 'Air Filter', 'vendors': [{'name': 'Vendor P1', 'price': 16.00}, {'name': 'Vendor Q1', 'price': 16.50}]},
        {'part': 'AC Filter', 'vendors': [{'name': 'Vendor R1', 'price': 17.00}, {'name': 'Vendor S1', 'price': 17.50}]},
        {'part': 'Oil Filter', 'vendors': [{'name': 'Vendor T1', 'price': 18.50}, {'name': 'Vendor U1', 'price': 19.00}]}
    ]
}

# Load the trained model
model = load_model('vehicle_model.h5')

# Function to classify the uploaded image
def classify_image(image):
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

# Function to get recommended parts with vendor-specific prices
def get_recommended_parts(vehicle):
    return recommended_parts.get(vehicle, [])

@app.route('/classify', methods=['POST'])
def classify_vehicle():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Classify the image
        predicted_class, confidence = classify_image(file)
        parts = get_recommended_parts(predicted_class)
        
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'recommended_parts': parts
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
