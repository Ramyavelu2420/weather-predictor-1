from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [
            float(request.form['temperature_C']),
            float(request.form['pressure_kpa']),
            float(request.form['relative_humidity']),
            float(request.form['wind_speed_kmph']),
            float(request.form['visibility_km']),
            int(request.form['hour'])
        ]
        
        # Prepare and scale features
        final_features = scaler.transform(np.array(features).reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        result = "Sunny" if prediction == 0 else "Rainy"
        
        return render_template('index.html', 
                             prediction_text=f'Predicted Weather: {result}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
