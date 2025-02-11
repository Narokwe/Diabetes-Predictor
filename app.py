from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize Flask app and set template folder to 'templates'
app = Flask(__name__, template_folder='templates')  # Explicitly set the templates folder

# Debugging: Print statements to check loading status
print("Loading model...")
model = joblib.load('diabetes_model.pkl')
print("Model loaded.")

print("Loading scaler...")
scaler = joblib.load('scaler.pkl')
print("Scaler loaded.")

@app.route('/')
def home():
    print("Loading index.html...")  # Add this print statement to check if this function is reached
    return render_template('index.html')  # This will load the HTML page from the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form and ensure correct feature order
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])
        
        # Create a numpy array of the features
        data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        
        # Scale the input data
        data = scaler.transform(data)  # Apply scaling to the input data
        
        # Predict diabetes using the model
        prediction = model.predict(data)  # Predict diabetes using the model
        
        # Return prediction result as JSON
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return jsonify({'prediction': result})

    except Exception as e:
        # Handle any errors and return them as JSON
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  