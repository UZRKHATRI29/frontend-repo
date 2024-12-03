from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS after initializing the Flask app

data = pd.read_csv('Banglore_cleaned_data.csv')
pipe = pickle.load(open("RigeModel.pkl", 'rb'))  # Ensure correct relative path

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route("/predict", methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    
    # Create input DataFrame
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0]
    
    # Return prediction as string multiplied to represent value
    return str(np.round(prediction, 2) * 100000)

if __name__ == "__main__":
    app.run(debug=False, port=5500)  # Set debug to False for production
