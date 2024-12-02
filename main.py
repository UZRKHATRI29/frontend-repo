from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

data= pd.read_csv('Banglore_cleaned_data.csv')
pipe = pickle.load(open("E:\Ededge\Projects\House Prediction\RigeModel.pkl", 'rb'))

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
    
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0]
    
    return str(np.round(prediction, 2) * 100000) # Convert to string before returning


if __name__ == "__main__":
    app.run(debug=True, port=5500)