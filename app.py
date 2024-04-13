from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import base64
import cv2

Image_size = 100
# Create application
app = Flask(__name__, template_folder="templates",static_folder="static",static_url_path="/")
# Load the model
model = load_model('./preparingmodel/Model.h5')



# Create Home url
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html', prediction_result = False)

# Create classifier api
@app.route('/predict', methods = ['POST'])
def predict():
    # Get the uploaded image from the request
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Resize the image to match the model's input size
    resized_image = cv2.resize(image, (Image_size, Image_size))
    # Normalize pixel values to the range [0, 1]
    preprocessed_image = resized_image / 255.0
    # Make prediction using your model
    probability_result = model.predict(np.expand_dims(preprocessed_image, axis=0))
    # Process the prediction result
    predicted_class = ['Masked','Not Masked'][np.argmax(probability_result)]
    # Encode the image to enable us to sent to html file
    encoded_image = base64.b64encode(cv2.imencode('.jpg', resized_image)[1]).decode('utf-8')
    return render_template('index.html',prediction_result=predicted_class, myimg=encoded_image)


# Run the application
if __name__ == "__main__":
    app.run(debug=True)


