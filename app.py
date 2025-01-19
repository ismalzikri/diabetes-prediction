from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form[field]) for field in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]
        # Reshape data for prediction
        input_data = np.array(data).reshape(1, -1)
        # Make prediction
        prediction = model.predict(input_data)
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
