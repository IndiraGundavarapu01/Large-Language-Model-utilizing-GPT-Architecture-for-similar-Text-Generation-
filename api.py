from flask import Flask, request, jsonify
import torch
import pickle
from model import GPTLanguageModel 

app = Flask(__name__)

# Load the PyTorch model
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        # Preprocess data if needed
        # Make predictions using the loaded model
        result = model.predict(data)  # Replace with your prediction code
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
