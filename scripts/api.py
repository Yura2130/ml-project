# api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Загрузка модели
model_path = 'models/random_forest_model.pkl'
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        
        # Предсказание
        prediction = model.predict(df)
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)